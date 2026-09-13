# Decision Support V1

Decision Support V1 is the first layer after live forecasting and live diagnosis. Its purpose is to turn fresh, already-computed evidence into transparent operator recommendations **without controlling equipment**.

The runtime chain is:

```text
read-only OPC-UA runtime
        ↓
Live Forecasting ─────┐
                     ├─→ Decision Support V1 → ranked advisory scenarios → operator
Live Diagnosis ──────┘
```

The boundary is intentional. Decision Support V1 is not an optimizer, process simulator, causal controller, LLM agent, or PLC command layer.

## What V1 evaluates

A `DecisionPolicy` contains three transparent objects:

- **Evidence conditions** inspect a small frozen vocabulary of forecasting or diagnosis evidence;
- **Recommendation scenarios** describe operator-visible candidate actions or operating options;
- **Decision rules** add deterministic points to scenarios when conditions match.

Supported forecast metrics are `min`, `max`, `mean`, `last`, `lower_min`, and `upper_max` for one named forecast target. Supported diagnosis metrics are `severity`, `ratio_to_threshold`, `trailing_anomaly_points`, and `latest_anomaly`.

A scenario may also define guardrails. If any guardrail is unavailable or false, the scenario is ineligible. Missing forecast intervals therefore do not silently satisfy rules that require `lower_min` or `upper_max`.

The final ordering is deterministic:

```text
scenario score = base priority + sum(points from matched rules)
```

Eligible scenarios whose score is at least `minimum_recommendation_score` are ranked by score and then by scenario id for a deterministic tie break.

## Evidence freshness and binding

`DecisionSupportRegistry` binds one decision-support application to exactly one forecasting application and one diagnosis application.

Registration and every evaluation verify that both evidence applications share the same:

- workspace project;
- data source;
- runtime id.

Each evaluation requests fresh evidence from both registries. Their own live services continue to enforce the active-runtime and stale-buffer checks already frozen in Live Forecasting V1 and Live Diagnosis V1.

Decision Support V1 additionally checks the two `observed_through` timestamps. If their absolute skew exceeds the policy's `max_observation_skew_seconds`, evaluation fails closed instead of ranking scenarios from temporally mismatched evidence.

## Scenario semantics

A scenario's `advisory_parameters` are **descriptive values for operator review**, not a write payload. For example:

```json
{
  "scenario_id": "review-derate-10",
  "title": "Review a 10% load reduction",
  "advisory_parameters": {
    "load_reduction_pct": 10
  }
}
```

The runtime does not map this object to OPC-UA nodes, setpoints, PLC commands, actuators, or any other execution surface.

Decision Support V1 also does not claim that a scenario will produce a quantified physical effect. The result explicitly reports:

```text
scenario_evaluation_scope = deterministic_policy_rule_evaluation_not_process_simulation
claim_scope = advisory_evidence_ranking_not_causal_or_control
```

A later optimization or digital-twin layer may provide modeled counterfactual outcomes, but that is outside V1.

## API

The live application exposes:

- `GET /v1/decision-support/capabilities`
- `POST /v1/projects/{project_id}/decision-support/applications`
- `GET /v1/decision-support/applications`
- `GET /v1/decision-support/applications/{application_id}`
- `POST /v1/decision-support/applications/{application_id}/evaluate`
- `DELETE /v1/decision-support/applications/{application_id}`

Policy registration persists a transparent JSON binding in the workspace. No training, calibration, model selection, online learning, or optimization is performed during registration.

## Frozen safety properties

Decision Support V1 is frozen around these properties:

- `advisory_only=true`;
- `operator_confirmation_required=true`;
- `execution_enabled=false`;
- `control_actions_enabled=false`;
- `opcua_writes_enabled=false`;
- `online_learning=false`;
- `llm_actions_enabled=false`;
- production decision-support modules contain no OPC-UA write primitive and no model-fitting call;
- the underlying production OPC-UA client remains browse/read/subscribe only.

The local asyncua smoke server may write its own test variables to simulate changing PLC telemetry. Those writes are fixture-side only and are not exposed through the product runtime.

## Deliberately out of scope

Decision Support V1 does not include:

- PLC or OPC-UA writes;
- automatic execution of recommendations;
- optimization solvers, MILP, MPC, or closed-loop control;
- learned action-effect models;
- process simulation or digital-twin counterfactuals;
- causal root-cause or causal action-effect claims;
- online policy learning;
- LLM/Agent action execution;
- MQTT expansion.

This creates a clean product boundary: IndusTSFM can already acquire live data, forecast, diagnose, and rank transparent operator recommendations while remaining read-only at the OT interface.
