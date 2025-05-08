from gymnasium.envs.registration import register



# — Anchor expansion probe —
register(
    id="AnchorExpEnv-v0",
    entry_point="simulation.envs:AnchorExpEnv",
    max_episode_steps=200,
)
