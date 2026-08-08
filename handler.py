import runpod

from runpod_swinir.worker import handler


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
