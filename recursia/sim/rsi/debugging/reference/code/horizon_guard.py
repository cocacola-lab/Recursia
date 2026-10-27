"""Harness-only bounded episode execution, identical for parent and candidates."""


class EpisodeHorizonReached(BaseException):
    """Expected control-flow stop; ordinary policy exceptions are never caught."""


class HorizonGuard:
    def __init__(self, low, limit=4000):
        assert low.max_steps == limit
        self.low = low
        self.limit = limit
        self.exhausted = False
        self.forwarded_steps = 0
        original = low.handle.step

        def step(action):
            if self.exhausted or low._sim_step_count >= limit:
                self.exhausted = True
                raise EpisodeHorizonReached()
            before = low._sim_step_count
            result = original(action)
            self.forwarded_steps += 1
            if before + 1 >= limit:
                # The interrupted low-level caller normally commits these fields
                # after handle.step returns. Commit its final state exactly once.
                low._current_obs, low._current_reward, low._current_done, low._current_info = result
                low._sim_step_count = before + 1
                self.exhausted = True
                raise EpisodeHorizonReached()
            return result

        low.handle.step = step

    def block(self, code, namespace):
        namespace['_rsi_expected_horizon'] = EpisodeHorizonReached
        namespace['_rsi_horizon_guard'] = self
        return ("try:\n    exec(compile(" + repr(code) + ", '<frozen_policy>', 'exec'), globals(), globals())\n"
                "except _rsi_expected_horizon:\n"
                "    if not _rsi_horizon_guard.exhausted:\n        raise\n"
                "    print('EPISODE_TIMEOUT horizon=4000')\n")


def eligible_timeout_count(count, total):
    return total > 0 and 100 * count <= 2 * total
