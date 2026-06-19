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

- `entrypoint` role=`entrypoint` degree=`110` file=`` line=``
- `parser` role=`entrypoint` degree=`51` file=`arch_builder/train_vertical_slice.py` line=`919`
- `tools/agent_codegraph/build_agent_codegraph.py` role=`scanner` degree=`43` file=`` line=``
- `controller` role=`controller` degree=`39` file=`` line=``
- `memory` role=`memory` degree=`38` file=`` line=``
- `parser` role=`entrypoint` degree=`37` file=`tools/project_probe/probe_learning_loop.py` line=`60`
- `ActionMatrixLayer.__init__` role=`controller` degree=`35` file=`arch_builder/model.py` line=`26`
- `tools/build_code_logic_graph.py` role=`scanner` degree=`35` file=`` line=``
- `tools/agent_inspector/inspect_project.py` role=`memory` degree=`32` file=`` line=``
- `arch_builder/train_vertical_slice.py` role=`data` degree=`30` file=`` line=``
- `ActionMatrixLayer.forward` role=`controller` degree=`29` file=`arch_builder/model.py` line=`98`
- `tools/project_probe/forced_program_oracle.py` role=`entrypoint` degree=`28` file=`` line=``
- `ActionMatrixModel.__init__` role=`model_core` degree=`24` file=`arch_builder/model.py` line=`249`
- `tools/project_probe/probe_learning_loop.py` role=`entrypoint` degree=`24` file=`` line=``
- `scanner` role=`scanner` degree=`23` file=`` line=``
- `HybridScanner.__init__` role=`scanner` degree=`22` file=`arch_builder/hybrid_scanner.py` line=`14`
- `train` role=`reporting` degree=`22` file=`arch_builder/train_vertical_slice.py` line=`772`
- `arch_builder/model.py` role=`controller` degree=`21` file=`` line=``
- `FuncVisitor._target` role=`memory` degree=`21` file=`tools/agent_codegraph/build_agent_codegraph.py` line=`188`
- `FV._target` role=`memory` degree=`21` file=`tools/agent_inspector/inspect_project.py` line=`84`
- `FuncVisitor._target` role=`memory` degree=`21` file=`tools/build_code_logic_graph.py` line=`133`
- `primitive_matrix` role=`primitive_matrix` degree=`19` file=`` line=``
- `ActionExecutor.__init__` role=`executor` degree=`18` file=`arch_builder/executor.py` line=`12`
- `arch_builder/primitive_matrix.py` role=`primitive_matrix` degree=`18` file=`` line=``
- `PrimitiveMatrix5x5.__init__` role=`primitive_matrix` degree=`18` file=`arch_builder/primitive_matrix.py` line=`63`

## Probe metrics

- `loss`: `9.03807544708252`
- `train_acc`: `0.375`
- `gain_disabled_delta`: `-0.0031250715255737305`
- `sim_result_disabled_delta`: `-0.006227254867553711`
- `sim_disabled_delta`: `-0.008959949016571045`
- `choice_without_sim_delta`: `0.03444105014204979`
- `ce_loss`: `0.7542140483856201`
- `sim_loss`: `0.14550462365150452`
- `expected_choice_loss`: `3.3086817264556885`
- `non_expected_primitive_loss`: `0.19061166048049927`
- `expected_active_loss`: `1.5253894329071045`
- `non_expected_active_loss`: `0.24604131281375885`
- `non_expected_tape_loss`: `0.0752803385257721`
- `non_expected_transform_loss`: `0.5465383529663086`
- `primitive_usage_diversity`: `0.0`
- `cell_choice_diversity`: `0.010960831306874752`
- `active_budget`: `0.007639226503670216`
- `tape_budget`: `0.0036056602839380503`
- `layer_action_diversity`: `0.0`
- `min_transform_loss`: `0.0`
- `signal_expected_choice_mass`: `0.11993330717086792`
- `signal_candidate_present`: `1.0`
- `signal_primitive_top_share`: `0.2155209481716156`
- `signal_active_mean`: `0.24490505456924438`
- `signal_tape_mean`: `0.07502328604459763`
- `signal_active_cells_soft`: `3.9184811115264893`
- `adaptive_recovery_gate`: `0.015893349424004555`
- `adaptive_collapse_gate`: `0.00023924781999085099`
- `adaptive_sparse_gate`: `0.010306360200047493`
- `adaptive_choice_boost`: `2.4761600494384766`
- `eff_lambda_choice`: `2.4761600494384766`
- `eff_lambda_non_expected_primitive`: `5.9811954997712746e-05`
- `eff_lambda_primitive_usage_diversity`: `1.196239099954255e-05`
- `eff_lambda_cell_choice_diversity`: `1.196239099954255e-05`
- `eff_lambda_non_expected_active`: `0.00020612719526980072`
- `eff_lambda_non_expected_tape`: `0.0005153180100023746`
- `eff_lambda_non_expected_transform`: `0.00020612719526980072`
- `eff_lambda_active_budget`: `0.00020612719526980072`
- `eff_lambda_tape_budget`: `0.00020612719526980072`
- `eff_lambda_layer_action_diversity`: `1.196239099954255e-05`
- `program_recovery_rate`: `0.125`
- `expected_edge_recovery`: `0.125`
- `expected_any_recovery`: `0.203125`
- `expected_candidate_present`: `1.0`
- `expected_edge_choice_mass`: `0.11993330717086792`
- `expected_edge_active`: `0.2189171016216278`
- `transform_mass`: `0.5470783710479736`
- `skip_mass`: `0.2692659795284271`
- `disable_mass`: `0.18365570902824402`
- `choice_entropy`: `1.4590083360671997`
- `active_cells`: `16.0`
- `expected_top_cells`: `6.0`
- `primitive_top_share`: `0.2155209556221962`
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
- `usage_entropy`: `0.4784822165966034`
- `action_L0_0_1_diff_present`: `1.0`
- `action_L0_0_1_diff_choice_mass`: `0.11993330717086792`
- `action_L0_0_1_diff_recovery`: `0.125`
- `action_L0_0_1_diff_active`: `0.2189171016216278`
- `action_L0_0_1_diff_tape`: `0.06900455057621002`
- `semantic_grid_mismatch`: `0.3857421875`
- `grid_candidate_usage`: `0.22245435416698456`
- `semantic_candidate_usage`: `0.2441367357969284`
- `usage_candidate_usage`: `0.42305320501327515`
- `random_candidate_usage`: `0.11035570129752159`
- `scanner_source_mass_sum`: `0.9999999962747097`
- `edge_scale_mean`: `0.991568386554718`
- `cell_output_gate_mean`: `0.5586023330688477`
- `cell_tape_weight_mean`: `0.07502328604459763`
- `cell_write_mass_mean`: `0.20062002539634705`
- `target_write_gate_mean`: `0.582230269908905`
- `edge_pair_bias_expected`: `0.0`
- `edge_pair_bias_std`: `0.0`
- `write_pair_bias_expected`: `0.0`
- `write_pair_bias_std`: `0.0`
- `phase_pair_bias_expected`: `0.0`
- `phase_pair_bias_std`: `0.0`
- `cell_output_pair_bias_expected`: `0.0`
- `cell_output_pair_bias_std`: `0.0`
- `loss_accounting_error`: `1.334045833800701e-07`

## Loss connectivity

- `ce_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'choice_controller': 0.05884862749501821, 'gate_controller': 0.17221085014330978, 'scanner': 0.04348081417719334, 'simulator': 0.060821465036172286, 'executor': 0.019425587852419567, 'classifier': 0.48993003776062155}`
- `sim_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'simulator': 0.40144985809240963}`
- `expected_choice_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'choice_controller': 11.136004402305526, 'scanner': 11.06201486855291, 'simulator': 30.420345299923383}`
- `non_expected_primitive_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'choice_controller': 0.47309217492381783, 'scanner': 0.9789131078304091, 'simulator': 3.3050557837803987}`
- `expected_active_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'gate_controller': 4.761424932500316}`
- `non_expected_active_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'gate_controller': 0.2577312808491464}`
- `non_expected_tape_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'gate_controller': 0.11549900156152165}`
- `non_expected_transform_loss`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'gate_controller': 0.45246352996584216}`
- `primitive_usage_diversity`: requires_grad=`True` applicable=`True` connected=`False` target_grad_norms=`{'choice_controller': 0.0, 'scanner': 0.0, 'simulator': 0.0}`
- `cell_choice_diversity`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'choice_controller': 0.021186203557206674, 'scanner': 0.04484954389347954, 'simulator': 0.08474400228745255}`
- `active_budget`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'gate_controller': 0.07007441006616898}`
- `tape_budget`: requires_grad=`True` applicable=`True` connected=`True` target_grad_norms=`{'gate_controller': 0.01398009601464308}`
- `layer_action_diversity`: requires_grad=`True` applicable=`True` connected=`False` target_grad_norms=`{'choice_controller': 0.0, 'scanner': 0.0, 'simulator': 0.0}`
- `min_transform_loss`: requires_grad=`True` applicable=`True` connected=`False` target_grad_norms=`{'gate_controller': 0.0}`
