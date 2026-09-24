from verl.utils.reward_score.livecodebench.lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from verl.utils.reward_score.livecodebench.lcb_runner.evaluation.pass_k_utils import extract_instance_results
from verl.utils.reward_score.livecodebench.lcb_runner.runner.scenario_router import Scenario, sort_and_extract_save_results
from verl.utils.reward_score.livecodebench.lcb_runner.utils.extraction_utils import extract_code


def run_unit_test(benchmark, combined_results, num_process_evaluate=12, timeout=6, complete_evaluation=False):
    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    generations = [extracted for _, extracted in combined_results]
    info_dict = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
        complete_evaluation=complete_evaluation,
    )
    info_dict["graded"] = extract_instance_results(info_dict["results"])[0][0]
    return info_dict


def lcb_compute_score(custom_outputs, benchmark, timeout=6, complete_evaluation=False):
    """
    LiveCodeBench evaluation
    """
    # Extract the code from the outputs
    for custom_output in custom_outputs:
        custom_output["code_list"] = [
            extract_code(output, None) for output in custom_output["output_list"]
        ]

    # Check if all code strings are empty
    for custom_output in custom_outputs:
        if all(len(code.strip()) == 0 for code in custom_output["code_list"]):
            return {"graded": 0, "metadata": "No code found"}

    # Assert the format of the outputs
    assert len(custom_outputs) == len(benchmark), (
        f"{len(custom_outputs)} != {len(benchmark)}"
    )
    custom_outputs = [
        custom_output["code_list"]
        for custom_output in sorted(
            custom_outputs, key=lambda x: str(x["question_id"])
        )
    ]

    # Insert the outputs into the benchmark
    save_results = [
        instance.insert_output(custom_output, custom_output)
        for instance, custom_output in zip(benchmark, custom_outputs)
    ]
    save_results, combined_results = sort_and_extract_save_results(
        Scenario.codegeneration, save_results
    )

    # Run unit test
    info_dict = run_unit_test(benchmark, combined_results,
                              timeout=timeout, complete_evaluation=complete_evaluation)
    return info_dict
