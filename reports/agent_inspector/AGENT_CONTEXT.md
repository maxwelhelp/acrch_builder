# AGENT_CONTEXT

## Runtime truth

- gradient_closed: `True` bad=`[]`
- credit_closed: `True` bad=`[]` broken=`[]`

- recovery_loss_connected: `True`
- detached_enabled_losses: `[]`

## Static loops

- `main_learning_loop` static_closed=`True` missing_roles=`[]` missing_edges=`[]`
- `simulator_choice_loop` static_closed=`True` missing_roles=`[]` missing_edges=`[]`
- `memory_state_loop` static_closed=`True` missing_roles=`[]` missing_edges=`[]`
- `report_agent_loop` static_closed=`True` missing_roles=`[]` missing_edges=`[]`

## Risk nodes

- `entrypoint` role=`entrypoint` degree=`101` file=`` line=``
- `parser` role=`entrypoint` degree=`51` file=`arch_builder/train_vertical_slice.py` line=`904`
- `tools/agent_codegraph/build_agent_codegraph.py` role=`scanner` degree=`43` file=`` line=``
- `controller` role=`controller` degree=`38` file=`` line=``
- `parser` role=`entrypoint` degree=`37` file=`tools/project_probe/probe_learning_loop.py` line=`60`
- `memory` role=`memory` degree=`36` file=`` line=``
- `ActionMatrixLayer.__init__` role=`controller` degree=`35` file=`arch_builder/model.py` line=`26`
- `tools/build_code_logic_graph.py` role=`scanner` degree=`35` file=`` line=``
- `tools/agent_inspector/inspect_project.py` role=`memory` degree=`32` file=`` line=``
- `arch_builder/train_vertical_slice.py` role=`data` degree=`30` file=`` line=``
- `ActionMatrixLayer.forward` role=`controller` degree=`29` file=`arch_builder/model.py` line=`98`
- `tools/project_probe/forced_program_oracle.py` role=`entrypoint` degree=`28` file=`` line=``
- `ActionMatrixModel.__init__` role=`model_core` degree=`24` file=`arch_builder/model.py` line=`249`
- `tools/project_probe/probe_learning_loop.py` role=`entrypoint` degree=`24` file=`` line=``
- `HybridScanner.__init__` role=`scanner` degree=`22` file=`arch_builder/hybrid_scanner.py` line=`14`
- `train` role=`reporting` degree=`22` file=`arch_builder/train_vertical_slice.py` line=`757`
- `arch_builder/model.py` role=`controller` degree=`21` file=`` line=``
- `FuncVisitor._target` role=`memory` degree=`21` file=`tools/agent_codegraph/build_agent_codegraph.py` line=`188`
- `FV._target` role=`memory` degree=`21` file=`tools/agent_inspector/inspect_project.py` line=`84`
- `FuncVisitor._target` role=`memory` degree=`21` file=`tools/build_code_logic_graph.py` line=`133`
- `scanner` role=`scanner` degree=`19` file=`` line=``
- `ActionExecutor.__init__` role=`executor` degree=`18` file=`arch_builder/executor.py` line=`12`
- `arch_builder/primitive_matrix.py` role=`primitive_matrix` degree=`18` file=`` line=``
- `PrimitiveMatrix5x5.__init__` role=`primitive_matrix` degree=`18` file=`arch_builder/primitive_matrix.py` line=`63`
- `build_graph` role=`memory` degree=`18` file=`tools/agent_codegraph/build_agent_codegraph.py` line=`240`

## Probe metrics

- `loss`: `6.961708068847656`
- `train_acc`: `0.125`
- `gain_disabled_delta`: `-0.006433546543121338`
- `sim_result_disabled_delta`: `0.0003304481506347656`
- `sim_disabled_delta`: `-0.005459249019622803`
- `choice_without_sim_delta`: `0.03723834082484245`
- `ce_loss`: `0.7869494557380676`
- `sim_loss`: `0.13663145899772644`
- `expected_choice_loss`: `2.5705406665802`
- `non_expected_primitive_loss`: `0.19173772633075714`
- `expected_active_loss`: `1.6482962369918823`
- `non_expected_active_loss`: `0.2256380319595337`
- `non_expected_tape_loss`: `0.06470074504613876`
- `non_expected_transform_loss`: `0.5253682136535645`
- `primitive_usage_diversity`: `0.0`
- `cell_choice_diversity`: `0.010454756207764149`
- `active_budget`: `0.0021053499076515436`
- `tape_budget`: `0.002501905430108309`
- `layer_action_diversity`: `0.0`
- `min_transform_loss`: `0.0`
- `signal_expected_choice_mass`: `0.26534172892570496`
- `signal_candidate_present`: `1.0`
- `signal_primitive_top_share`: `0.22199401259422302`
- `signal_active_mean`: `0.22459672391414642`
- `signal_tape_mean`: `0.06473812460899353`
- `signal_active_cells_soft`: `3.593547821044922`
- `adaptive_recovery_gate`: `0.09044378995895386`
- `adaptive_collapse_gate`: `0.0014743506908416748`
- `adaptive_sparse_gate`: `0.05405407026410103`
- `adaptive_choice_boost`: `2.3643343448638916`
- `eff_lambda_choice`: `2.3643343448638916`
- `eff_lambda_non_expected_primitive`: `0.0003685876727104187`
- `eff_lambda_primitive_usage_diversity`: `7.371753599727526e-05`
- `eff_lambda_cell_choice_diversity`: `7.371753599727526e-05`
- `eff_lambda_non_expected_active`: `0.0010810813400894403`
- `eff_lambda_non_expected_tape`: `0.0027027034666389227`
- `eff_lambda_non_expected_transform`: `0.0010810813400894403`
- `eff_lambda_active_budget`: `0.0010810813400894403`
- `eff_lambda_tape_budget`: `0.0010810813400894403`
- `eff_lambda_layer_action_diversity`: `7.371753599727526e-05`
- `program_recovery_rate`: `0.375`
- `expected_edge_recovery`: `0.375`
- `expected_any_recovery`: `0.2109375`
- `expected_candidate_present`: `1.0`
- `expected_edge_choice_mass`: `0.26534172892570496`
- `expected_edge_active`: `0.20415031909942627`
- `transform_mass`: `0.5282174050807953`
- `skip_mass`: `0.29092539846897125`
- `disable_mass`: `0.18085718154907227`
- `choice_entropy`: `1.4595040082931519`
- `active_cells`: `16.0`
- `expected_top_cells`: `8.0`
- `primitive_top_share`: `0.22199401259422302`
- `active_edges_per_target`: `4.0`
- `final_read_last`: `1.0`
- `final_read_mean`: `0.0`
- `final_read_learned`: `0.0`
- `state_norm_none`: `1.0`
- `state_norm_layernorm`: `0.0`
- `slot_address_used_by_controller`: `1.0`
- `slot_address_used_by_executor`: `0.0`
- `primitive_embedding_rank`: `25.0`
- `primitive_pair_cos_mean`: `0.32978546619415283`
- `primitive_pair_cos_max`: `0.9889096617698669`
- `usage_entropy`: `0.4713374376296997`
- `action_L0_0_1_diff_present`: `1.0`
- `action_L0_0_1_diff_choice_mass`: `0.26534172892570496`
- `action_L0_0_1_diff_recovery`: `0.375`
- `action_L0_0_1_diff_active`: `0.20415031909942627`
- `action_L0_0_1_diff_tape`: `0.07115307450294495`
- `semantic_grid_mismatch`: `0.400390625`
- `grid_candidate_usage`: `0.24252018332481384`
- `semantic_candidate_usage`: `0.21766570210456848`
- `usage_candidate_usage`: `0.4706409275531769`
- `random_candidate_usage`: `0.06917318655177951`
- `scanner_source_mass_sum`: `0.9999999995343387`
- `edge_scale_mean`: `1.008556067943573`
- `cell_output_gate_mean`: `0.5402731597423553`
- `cell_tape_weight_mean`: `0.06473812274634838`
- `cell_write_mass_mean`: `0.1852685660123825`
- `target_write_gate_mean`: `0.5458730459213257`
- `edge_pair_bias_expected`: `0.0`
- `edge_pair_bias_std`: `0.0`
- `write_pair_bias_expected`: `0.0`
- `write_pair_bias_std`: `0.0`
- `phase_pair_bias_expected`: `0.0`
- `phase_pair_bias_std`: `0.0`
- `cell_output_pair_bias_expected`: `0.0`
- `cell_output_pair_bias_std`: `0.0`
- `loss_accounting_error`: `1.8067987639369676e-07`

## Loss connectivity

- `ce_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'choice_controller': 0.04798786609329644, 'gate_controller': 0.20654294563502415, 'scanner': 0.03675945355314609, 'simulator': 0.0854364661876969, 'executor': 0.011391311034654123, 'classifier': 0.5945156073822719}`
- `sim_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'simulator': 0.4089199768146404}`
- `expected_choice_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'choice_controller': 6.928307996754425, 'scanner': 12.10381828003443, 'simulator': 28.278313555736958}`
- `non_expected_primitive_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'choice_controller': 0.48794930818634824, 'scanner': 1.4221591092968224, 'simulator': 3.7667408123058985}`
- `expected_active_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'gate_controller': 4.390616789518021}`
- `non_expected_active_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'gate_controller': 0.2660497623348922}`
- `non_expected_tape_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'gate_controller': 0.11466064964427364}`
- `non_expected_transform_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'gate_controller': 0.4960046617891794}`
- `primitive_usage_diversity`: requires_grad=`True` applicable=`True` connected=`False` target_grad_norms=`{'choice_controller': 0.0, 'scanner': 0.0, 'simulator': 0.0}`
- `cell_choice_diversity`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'choice_controller': 0.03244269996836158, 'scanner': 0.08777970212936301, 'simulator': 0.1597834331487735}`
- `active_budget`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'gate_controller': 0.028381592258114336}`
- `tape_budget`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'gate_controller': 0.012279669584669832}`
- `layer_action_diversity`: requires_grad=`True` applicable=`True` connected=`False` target_grad_norms=`{'choice_controller': 0.0, 'scanner': 0.0, 'simulator': 0.0}`
- `min_transform_loss`: requires_grad=`True` applicable=`True` connected=`False` target_grad_norms=`{'gate_controller': 0.0}`
