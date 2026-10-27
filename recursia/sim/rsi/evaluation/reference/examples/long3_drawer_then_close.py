"""Example orchestration program (hand-written reference, not LLM output; interpreter subset v2):
"put the black bowl in the bottom drawer of the cabinet and close it". Shows runtime branching on structured results."""


def run(context, stack):
    put = stack.subgoal('in_sliding', object='black bowl', reference='cabinet', part='drawer', side='bottom')
    close = stack.subgoal('close_articulated', part='drawer', side='bottom')
    p = stack.plan([put, close])
    if p['status'] != 'ok':
        stack.finish('unsupported', str(p['unsupported']))
        return
    r = stack.execute(put)
    if r['status'] != 'control_completed' and 'retry_subgoal' in r['allowed_recovery_actions']:
        stack.log('retrying placement', {'first': r['status'], 'reason': r['failure_reason']})
        r = stack.recover(r, 'retry_subgoal')
    if r['status'] != 'control_completed':
        if r['object_possibly_retained'] is not False:
            stack.recover(r, 'go_home')
            stack.finish('stopped', 'placement failed with the object possibly in hand: ' + str(r['failure_reason']))
            return
        seen = stack.locate('black bowl')
        stack.log('bowl after failed placement', {'found': seen['found'], 'centre': seen['centre']})
    c = stack.execute(close)
    if c['status'] != 'control_completed' and 'retry_subgoal' in c['allowed_recovery_actions']:
        c = stack.recover(c, 'retry_subgoal')
    done = r['status'] == 'control_completed' and c['status'] == 'control_completed'
    stack.finish('controls_done' if done else 'partial', 'place=' + r['status'] + ' close=' + c['status'])
