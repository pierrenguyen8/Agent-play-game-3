# Architecture

Production code lives only under `src/`.

```text
App / CLI
  -> runtime.AgentRuntime
      -> games.planet.perception.PlanetPerceptionBackend
      -> games.planet.state_builder.PlanetStateBuilder
      -> agent_core.policy.RuleBasedHighLevelPolicy
      -> agent_core.skills.SkillManager
      -> policies.low_level.LowLevelPolicy     (reuse-first seam)
      -> games.planet.PlanetAdapter
```

## Dependency direction

`agent_core` knows nothing about Planet.
`games/planet` may depend on `agent_core` and `interfaces`.
`runtime` composes modules but contains no training code.
`training`, `experiments`, and `tools` must never be imported by `src`.

This boundary is what keeps a future PyInstaller/Nuitka build small and predictable.
