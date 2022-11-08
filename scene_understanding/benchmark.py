import argparse
import habitat
import agents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="configs/hc_pointnav2021.test_scenes.rgbd.yaml")
    parser.add_argument("--agent", type=str, default="RandomAgent", choices=["RandomAgent"])
    parser.add_argument("--num-episodes", type=int, default=5)
    args = parser.parse_args()

    config_path = args.config_path

    config = habitat.get_config(config_path)
    agent_type = getattr(agents, args.agent)
    agent = agent_type(config=config)

    benchmark = habitat.Benchmark(config_path)
    avg_metrics = benchmark.local_evaluate(agent, args.num_episodes)
    print(avg_metrics)


if __name__ == "__main__":
    main()
