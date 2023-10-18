import argparse
import json
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None


async def parse_request(request: Request) -> dict:
    request_dict = await request.json()
    prompt = request_dict.pop("inputs")
    parameters = request_dict.pop("parameters")
    max_new_tokens = parameters.pop("max_new_tokens")
    return_full_text = parameters.pop("return_full_text", False)
    do_sample = parameters.pop("do_sample", True)

    sampling_params = SamplingParams(max_tokens=max_new_tokens,
                                     use_beam_search=not do_sample,
                                     **parameters)

    return {"prompt": prompt,
            "parameters": sampling_params,
            "return_full_text": return_full_text}


@app.post("/")
async def root(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """

    request_dict = await request.json()
    stream = request_dict.pop("stream", False)
    if stream:
        return await generate_stream(request)
    else:
        return await generate(request)


@app.post("/generate")
async def generate(request: Request) -> Response:
    request_id = random_uuid()
    query = await parse_request(request)
    results_generator = engine.generate(query["prompt"],
                                        query["parameters"],
                                        request_id)
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [
        prompt + output.text if query["return_full_text"] else output.text
        for output in final_output.outputs]
    ret = {"generated_text": text_outputs[0], "status": 200}
    return JSONResponse(ret)


@app.post("/generate_stream")
async def generate_stream(request: Request) -> Response:
    request_id = random_uuid()
    query = await parse_request(request)
    results_generator = engine.generate(query["prompt"],
                                        query["parameters"],
                                        request_id)

    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            #prompt = request_output.prompt
            #text_outputs = [
            #    prompt + output.text if query["return_full_text"] else output.text
            #    for output in request_output.outputs]
            output = request_output.outputs[0]
            ret = {"token":
                    {"text": output.text,
                     "id": request_output.request_id,
                     "logprob": output.cumulative_logprob,
                     "special": False}}
            yield (json.dumps(ret) + "\n").encode("utf-8")

    return StreamingResponse(stream_results())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
