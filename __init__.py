from gymnasium.envs.registration import register



# — Anchor expansion probe —
register(
    id="AnchorExpEnv-v0",
    entry_point="envs.AnchorExpEnv:AnchorExpEnv",
    max_episode_steps=200,
)