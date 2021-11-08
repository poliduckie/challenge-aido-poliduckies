import io
from typing import Optional

import numpy as np
from PIL import Image

from wrappers import SBWrapper

from aido_schemas import (
    Context,
    DB20Commands,
    DB20Observations,
    EpisodeStart,
    JPGImage,
    LEDSCommands,
    logger,
    no_hardware_GPU_available,
    protocol_agent_DB20,
    PWMCommands,
    RGB,
    wrap_direct,
)


class StableBaselineAgent:

    def __init__(self, load_model: bool, model_path: Optional[str]):
        self.load_model = load_model
        self.model_path = model_path

    def init(self, context: Context):
        self.check_gpu_available(context)
        logger.info("StableBaselineAgent init")
        from stable_baselines3 import PPO

        self.preprocessor = SBWrapper()
        if self.model_path:
            self.model = PPO.load(self.model_path)
        else:
            self.model = PPO.load("model.zip")
        self.current_image = np.zeros((640, 480, 3))

        logger.info("Model loaded.")

    def check_gpu_available(self, context: Context):
        import torch

        available = torch.cuda.is_available()
        context.info(f"torch.cuda.is_available = {available!r}")
        context.info("init()")
        if available:
            i = torch.cuda.current_device()
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(i)
            context.info(f"device {i} of {count}; name = {name!r}")
        else:
            no_hardware_GPU_available(context)

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    def on_received_observations(self, data: DB20Observations):
        camera: JPGImage = data.camera
        obs = jpg2rgb(camera.jpg_data)
        self.current_image = self.preprocessor.preprocess(obs)

    def compute_action(self, observation):
        observation = self.preprocessor.preprocess(observation)
        action, _ = self.model.predict(observation)
        return action.astype(float)

    def on_received_get_commands(self, context: Context):
        pwm_left, pwm_right = self.compute_action(self.current_image)

        pwm_left = float(np.clip(pwm_left, -1, +1))
        pwm_right = float(np.clip(pwm_right, -1, +1))

        grey = RGB(0.0, 0.0, 0.0)
        led_commands = LEDSCommands(grey, grey, grey, grey, grey)
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = DB20Commands(pwm_commands, led_commands)
        context.write("commands", commands)

    def finish(self, context: Context):
        context.info("finish()")


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """Reads JPG bytes as RGB"""

    im = Image.open(io.BytesIO(image_data))
    im = im.convert("RGB")
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data

def main():
    node = StableBaselineAgent(load_model=False, model_path=None)
    protocol = protocol_agent_DB20
    wrap_direct(node=node, protocol=protocol)


if __name__ == "__main__":
    main()