"""Preserve transparent-container resets and score insertion or verified no-op.

Run the category through ConverterBase.convert_all with the shared
SourceInstance/ConversionContext API; conversion is CPU-only.
"""
from __future__ import annotations


def create_evaluator(session, controller, config, output):
    from copy import deepcopy
    from .partial_occlusion import IdentificationPlacementEvaluator
    from worldleague.tasks.completion import progress
    from worldleague.authoring.io import write_json

    class StableInsideEvaluator(IdentificationPlacementEvaluator):
        @property
        def requires_done_declaration(self):
            return self.parameters.get('require_inside_declaration_and_done') is True

        def begin_episode(self, observation):
            self.initial_relation = self.measure()['destination_relation']
            self.terminal_declared = False
            self.terminal_finalized = False
            self.terminal_result = None
            return super().begin_episode(observation)

        def declare_done(self, observation):
            return self._finish(observation, declared=True)

        def finalize_episode(self, observation):
            return self._finish(observation, declared=False)

        def resume_episode(self):
            # Resume the same episode: keep its declarations and physical history.
            self.terminal_declared = False
            self.terminal_finalized = False
            self.terminal_result = None

        def _finish(self, observation, *, declared):
            if not self.active:
                raise RuntimeError('Containment confirmation requires an active episode.')
            if self.terminal_result is None:
                self.terminal_declared = declared
                self.terminal_finalized = True
                self.terminal_result = self.record(observation, action=None)
            return deepcopy(self.terminal_result)

        def record(self, observation, *, action=None):
            if not self.active:
                return None
            if self.terminal_result is not None:
                return deepcopy(self.terminal_result)
            available, stable = self._update()
            physical_complete = bool(stable and self.stable_steps >= self.parameters['release_stability_steps'])
            confirmed = not self.requires_done_declaration or (
                (self.answer == 'inside' or bool(self.parameters.get('terminal_qa'))) and self.terminal_declared)
            item = {'id': 'stable_inside', 'kind': 'terminal_condition',
                'completed': physical_complete and confirmed,
                'evaluation_validity': 'valid' if available else 'unavailable'}
            result = {'schema': 'private_stable_inside_v1', 'snapshot_id': observation.snapshot_id,
                'action_step': observation.action_step, 'completion_items': [item],
                'task_success': item['completed'] and item['evaluation_validity'] == 'valid',
                'initial_inside_native': self.initial_relation,
                'release_stable_steps': self.stable_steps, 'measurements': self.last,
                'evaluation_validity': 'complete' if available else 'unavailable'}
            if self.requires_done_declaration:
                result.update(physical_inside_complete=physical_complete,
                    inside_declaration_received=self.answer == 'inside',
                    answer_history=deepcopy(self.answer_history),
                    model_declared_done=self.terminal_declared,
                    evaluation_finalized=self.terminal_finalized)
                if not self.terminal_finalized and available:
                    result['evaluation_validity'] = 'pending_done'
            result['no_op_verified'] = bool(result['task_success'] and self.initial_relation is True
                                           and observation.action_step == self.start_step)
            result.update(progress(result))
            write_json(self.output / 'state.json', result)
            return result

    native = deepcopy(config)
    native['bindings']['target'] = config['bindings']['obj_small']
    native['bindings']['destination'] = config['bindings']['obj_container']
    return StableInsideEvaluator(session, controller, native, output)
