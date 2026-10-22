# Domain model

## Ontology

Nodes: site, zone, equipment, sensor, setpoint, derived_tag, alarm_definition,
loop. Edges: part_of, measures, controls, serves, depends_on, indicates, monitors,
located_in.

## Derived tags

Derived tags declare an expression, an input mapping, and a unit. Computation runs
only over aligned timestamps. Missing inputs produce no sample.

## Operational math

- `tons = design_tons * load_pct / 100`
- `kW/ton = power_kw / tons`
- `COP = tons * 3.5168525 / power_kw`
- `delta-T = return_temp - supply_temp`
- `approach proxy = condenser_leaving_water - chw_supply`

The approach tag is a proxy because saturated condenser temperature is not in the
reference dataset; production should use the real approach.

## Alarms

Alarms carry severity, code, tag, value, limit, and lifecycle timestamps. Triage
clusters repeated codes within a window and ranks by severity, asset criticality,
repeat count, and age.
