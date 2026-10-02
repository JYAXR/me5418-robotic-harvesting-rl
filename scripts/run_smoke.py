"""Run a short random-policy smoke test."""

from me5418_harvesting import HarvestingEnv


def main() -> None:
    env = HarvestingEnv(render_mode="ansi")
    observation, info = env.reset(seed=0)
    print(f"observation_shape={observation.shape} reset_info={info}")
    total_reward = 0.0
    for _ in range(10):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        print(env.render())
        if terminated or truncated:
            break
    print(f"total_reward={total_reward:.3f} final_info={info}")


if __name__ == "__main__":
    main()

