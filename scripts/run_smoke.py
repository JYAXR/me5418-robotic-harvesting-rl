"""运行一次简短的随机策略冒烟测试。"""

from me5418_harvesting import HarvestingEnv


def main() -> None:
    env = HarvestingEnv(render_mode="ansi")
    observation, info = env.reset(seed=0)
    print(f"观测形状={observation.shape} 重置信息={info}")
    total_reward = 0.0
    for _ in range(10):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        print(env.render())
        if terminated or truncated:
            break
    print(f"累计奖励={total_reward:.3f} 最终信息={info}")


if __name__ == "__main__":
    main()
