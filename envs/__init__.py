from gymnasium.envs.registration import register

# — Cone penetration probe —
register(
    id="ConePenEnv-v0",
    entry_point="envs.ConePen_env:ConePenEnv",
    max_episode_steps=200,
)

# — Anchor expansion probe —
register(
    id="AnchorExpEnv-v0",
    entry_point="envs.AnchorExpEnv:AnchorExpEnv",
    max_episode_steps=200,
)
