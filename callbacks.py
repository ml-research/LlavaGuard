import os

from rtpt import rtpt
from transformers import TrainerCallback


class LlavaGuardCallback(TrainerCallback):
    "A callback that updates remaining time using rtpt"

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        model_name = 'LlavaGuard'
        # save state to output dir
        os.makedirs('output/tmp', exist_ok=True)
        state.save_to_json(json_path='output/tmp/state.json')
        print(f'current step: {state.global_step}, max steps: {state.max_steps}')
        self.r = rtpt.RTPT(name_initials='LH', experiment_name=f'{model_name}-Trainer',
                           max_iterations=state.max_steps - state.global_step)
        self.r.start()

    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        self.r.step()

    # def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     if rtpt not initialized, initialize it
    #     model_name = 'LlavaGuard'
    #     if not hasattr(self, 'r'):
    #         self.r = rtpt.RTPT(name_initials='LH', experiment_name=f'{model_name}-eval',
    #                            max_iterations=1)
    #         self.r.start()
    #     return control