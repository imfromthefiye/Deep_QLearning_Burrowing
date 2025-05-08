from gymnasium.envs.registration import register



# — Anchor expansion probe —
register(
    id="AnchorExpEnv-v0",
    entry_point="envs.AnchorExp_env:AnchorExpEnv",
    max_episode_steps=200,
)