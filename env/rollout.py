import numpy as np
import robosuite as rs

def main():
    env = rs.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )    
    
    env.reset()
    
    for i in range(50000):
        action = np.random.randn(*env.action_spec[0].shape) * 0.1
        obs, reward, done, info = env.step(action)
        env.render()
        print("Obersvation", obs)
        print("Reward", reward)


if __name__ == "__main__":
    main()
