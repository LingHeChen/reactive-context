"""Optional generative updater; separate from Jev's typed decisions."""

import json
from dataclasses import asdict

from pydantic_ai import Agent

from ..objective import WorkObjective


class PydanticObjectiveUpdater:
    def __init__(self, model, *, model_settings=None):
        self.last_result = None
        self.agent = Agent(
            model,
            output_type=WorkObjective,
            instructions=(
                "Maintain the current work objective using the supplied current state and "
                "recent evidence. Preserve still-applicable constraints and success criteria. "
                "Update the goal only for a user-requested change. Track next steps and status; "
                "do not write a narrative history summary or claim unsupported completion."
            ),
            model_settings=model_settings or {"max_tokens": 1024},
        )

    async def __call__(self, request):
        result = await self.agent.run(json.dumps(asdict(request), ensure_ascii=False))
        self.last_result = result
        return result.output
