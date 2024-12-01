initialize_environment(curriculum_stage)
initialize_agents()
initialize_central_critic()

for episode in range(total_episodes):
    env.reset(curriculum_stage)
    agent_states = env.get_initial_states()
    agent_types = initialize_agent_types(agent_states)
    done = False

    while not done:
        observations = {id: env.observe(id) for id in agent_states}

        for id in agent_states:
            if USE_CPD and check_type_change(id, observations[id]):
                agent_types[id] = infer_type(id, observations[id])
            elif not USE_CPD:
                agent_types[id] = infer_type(id, observations[id])

        joint_action_values = central_critic.evaluate(observations, agent_types)
        joint_actions = {id: select_action(id, joint_action_values) for id in agent_states}

        env.execute(joint_actions)
        env.update()
        rewards = env.get_rewards(joint_actions)

        for id in agent_states:
            update_policy(id, observations[id], agent_types[id], joint_actions[id], rewards[id])
        central_critic.update(observations, agent_types, joint_actions, rewards)

        agent_states = env.get_current_states()
        done = env.check_termination()

    if USE_CURRICULUM_LEARNING:
        curriculum_stage = adjust_curriculum(curriculum_stage)
