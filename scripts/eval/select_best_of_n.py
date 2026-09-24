# Eval.sh save the generated test cases evaluation results in the response file, so you can get the generated test cases evaluation results by get the metadata from the response file.

import random
import pandas as pd
import numpy as np
import argparse


def log_message(message, file=None, append=True):
    """
    Prints a message to console and writes it to a file if provided.

    Args:
        message: The message to print and write
        file: File object to write to (optional)
        append: Whether to add a newline (default: True)
    """
    print(message)
    if file:
        file.write(message + ('\n' if append else ''))


def get_valid_test_cases(solutions, metadata):
    """
    Returns a set of valid test case indices that are passed by at least 
    half of the solutions that pass public test cases.
    """

    # Find the maximum number of test cases in any solution
    max_test_cases = 0
    for sol_idx in solutions:
        results = metadata[sol_idx]['results'][0][0]
        max_test_cases = max(max_test_cases, len(results))

    valid_tests = set()
    # Check each test case
    for test_idx in range(max_test_cases):
        passing_count = 0
        for sol_idx in solutions:
            if sol_idx < len(metadata):
                results = metadata[sol_idx]['results'][0][0]
                if test_idx < len(results) and results[test_idx] == True:
                    passing_count += 1

        # Valid if at least half of solutions pass it
        if passing_count >= len(solutions) / 2:
            valid_tests.add(test_idx)

    return valid_tests


def get_public_private_scores(response_dataset):
    public_scores_list = []
    private_scores_list = []
    for i in range(len(response_dataset)):
        metadata = response_dataset.iloc[i].metadata
        public_test_cases_num = len(response_dataset.iloc[i].public_test_cases) if hasattr(
            response_dataset.iloc[i], 'public_test_cases') else 1
        public_scores = []
        private_scores = []
        for j in range(len(metadata)):
            # j-th response running result
            if isinstance(metadata[j], str):  # No think found
                public_scores.append(False)
                private_scores.append(False)
            else:
                assert isinstance(metadata[j], dict)
                # {"{'graded': 0, 'metadata': 'No code found'}"} or "error_message": "Error during testing
                if "results" not in metadata[j] or metadata[j]['results'][0][0][0] == -4:
                    public_scores.append(False)
                    private_scores.append(False)
                else:
                    results = metadata[j]['results'][0][0]
                    public_score = all(
                        [results[k] == True for k in range(min(public_test_cases_num, len(results)))])
                    private_score = all(
                        results[k] == True for k in range(len(results)))
                    public_scores.append(public_score)
                    private_scores.append(private_score)
        public_scores_list.append(public_scores)
        private_scores_list.append(private_scores)
    return public_scores_list, private_scores_list


def get_public_private_generated_right_index(response_dataset, validate_test_cases):
    # Pre-compute the public_case_true_idx and max_idx_list for each problem
    problem_indices = []
    vaild_filter = 0
    total_eliminated_public = 0  # Count eliminated public candidates
    total_missed_private = 0     # Count missed private candidates
    total_delta = 0.0  # Accumulate delta_rate between generate and public

    for i in range(len(response_dataset)):
        # Track best indices under two strategies:
        # 1) generate score among candidates that already pass public tests (public-filtered)
        # 2) generate score among ALL candidates (unfiltered)
        max_idx_public = []
        max_score_public = -1
        max_idx_all = []
        max_score_all = -1

        public_case_true_idx = []
        private_case_true_idx = []
        response_problem = response_dataset.iloc[i]

        # Find solutions that pass public and private test cases
        for j in range(len(response_problem['metadata'])):
            if response_problem['private_scores'][j] == True:
                private_case_true_idx.append(j)
            if response_problem['public_scores'][j] == True:
                public_case_true_idx.append(j)

        # Pre-compute valid test cases once for this problem
        valid_test_cases_set = None
        if validate_test_cases and len(public_case_true_idx) > 1:
            valid_test_cases_set = get_valid_test_cases(
                public_case_true_idx, response_problem['metadata'])

        # Calculate scores for each solution that passes public tests
        # Calculate generate scores for EVERY solution once to reuse for both strategies
        for j in range(len(response_problem['metadata'])):
            try:
                all_results = response_problem['generated_metadata'][j]['results'][0][0]
                # Calculate generate_score based on validation setting
                if validate_test_cases and valid_test_cases_set is not None:
                    # Count only valid test cases
                    generate_score = sum(1 for test_idx, result in enumerate(all_results)
                                         if result == True and test_idx in valid_test_cases_set)
                else:
                    # Original logic: count all passing test cases
                    generate_score = sum(
                        1 for result in all_results if result == True)
            except:
                generate_score = -2

            # --- unfiltered best ---
            if generate_score > max_score_all:
                max_score_all = generate_score
                max_idx_all = [j]
            elif generate_score == max_score_all:
                max_idx_all.append(j)

            # --- public-filtered best ---
            if j in public_case_true_idx:
                if generate_score > max_score_public:
                    max_score_public = generate_score
                    max_idx_public = [j]
                elif generate_score == max_score_public:
                    max_idx_public.append(j)

        # Check if public_case_true_idx and max_idx_list are different
        if set(public_case_true_idx) != set(max_idx_public):
            log_message(
                f"i: {i}, public_case_true_idx: {public_case_true_idx}, private_case_true_idx: {private_case_true_idx}, max_idx_list: {max_idx_public}", output_file)

            # Compute private pass rates for public vs generate selections
            pub_rate = sum(
                1 for idx in public_case_true_idx if response_problem['private_scores'][idx]) / len(public_case_true_idx)
            gen_rate = sum(
                1 for idx in max_idx_public if response_problem['private_scores'][idx]) / len(max_idx_public)
            delta_rate = gen_rate - pub_rate
            log_message(
                f"  public_rate: {pub_rate:.4f}, gen_rate: {gen_rate:.4f}, delta: {delta_rate:.4f}", output_file)
            total_delta += delta_rate

            # Calculate eliminated public candidates (in public but not in private or max)
            eliminated_public = [
                idx for idx in public_case_true_idx if idx not in private_case_true_idx and idx not in max_idx_public]
            total_eliminated_public += len(eliminated_public)
            if eliminated_public:
                log_message(
                    f"  eliminated_public: {eliminated_public}, count: {len(eliminated_public)}", output_file)

            # Check if there are elements in private_case_true_idx that are not in max_idx_list
            missed_private = [
                idx for idx in private_case_true_idx if idx not in max_idx_public]
            total_missed_private += len(missed_private)
            if missed_private:
                log_message(
                    f"  private_case_true_idx elements not in max_idx_list: {missed_private}, count: {len(missed_private)}", output_file)

            vaild_filter += 1

        problem_indices.append({
            'public_case_true_idx': public_case_true_idx,
            'private_case_true_idx': private_case_true_idx,
            'max_idx_list': max_idx_public,          # generate best among public-filtered
            'max_idx_all': max_idx_all             # generate best among ALL candidates
        })
    log_message(
        f"vaild_filter: {vaild_filter}, vaild_filter/len(response_dataset): {vaild_filter/len(response_dataset)}", output_file)
    log_message(
        f"total_eliminated_public: {total_eliminated_public}, total_missed_private: {total_missed_private}", output_file)
    log_message(
        f"total_delta: {total_delta:.4f}, average_delta: {total_delta/len(response_dataset):.4f}", output_file)
    public_failed_cases = 0
    for idx in range(len(problem_indices)):
        if len(problem_indices[idx]['public_case_true_idx']) == 0:
            public_failed_cases += 1
    log_message(
        f"public_failed_cases num is {public_failed_cases}, the rate is {public_failed_cases/len(problem_indices)}")
    return problem_indices


def calculate_analytical_pass_rates(response_dataset, problem_indices):
    """
    Calculates the analytical (expected) pass rates for different selection strategies.
    """
    N = len(response_dataset)
    if N == 0:
        return 0.0, 0.0, 0.0

    total_expected_pass_plain = 0.0
    total_expected_pass_public = 0.0
    total_expected_pass_generate = 0.0

    for i in range(N):
        response_problem = response_dataset.iloc[i]
        private_scores = response_problem['private_scores']
        num_solutions = len(private_scores)

        # 1. Plain Random Selection
        if num_solutions > 0:
            passed_private_count = sum(
                1 for score in private_scores if score == True)
            total_expected_pass_plain += passed_private_count / num_solutions

        indices_info = problem_indices[i]
        public_case_true_idx = indices_info['public_case_true_idx']
        max_idx_list = indices_info['max_idx_list']

        # 2. Public Test Selection
        if len(public_case_true_idx) > 0:
            passed_among_public = sum(
                1 for idx in public_case_true_idx if private_scores[idx] == True)
            total_expected_pass_public += passed_among_public / \
                len(public_case_true_idx)

        # 3. Generate Score Selection
        if len(max_idx_list) > 0:
            passed_among_max_score = sum(
                1 for idx in max_idx_list if private_scores[idx] == True)
            total_expected_pass_generate += passed_among_max_score / \
                len(max_idx_list)

    analytical_rate_plain = total_expected_pass_plain / N
    analytical_rate_public = total_expected_pass_public / N
    analytical_rate_generate = total_expected_pass_generate / N

    return analytical_rate_plain, analytical_rate_public, analytical_rate_generate


# ----------------- k-subset analysis -----------------

def _slice_first_k(lst, k):
    """Return first k elements of lst if lst is a list, else return lst unchanged."""
    if isinstance(lst, list):
        return lst[:k]
    return lst


def _simulate_and_log(df, problem_indices, ana_rates, n_trials, output_file, label):
    """
    Given a DataFrame `df` with 'private_scores', the precomputed `problem_indices`,
    and analytical rates `(plain, public, generate)`, run Monte Carlo for `n_trials`,
    compute stats, upper bound, and log all under `label`.
    """
    ana_plain, ana_public, ana_generate = ana_rates
    # Monte Carlo simulation
    plain_res, public_res, generate_res = [], [], []
    for _ in range(n_trials):
        cnt_plain = cnt_public = cnt_generate = 0
        for i, row in df.iterrows():
            priv = row['private_scores']
            if not priv:
                continue
            # plain random
            sel = random.randint(0, len(priv)-1)
            if priv[sel]:
                cnt_plain += 1
            # public selection
            pub_idx = problem_indices[i]['public_case_true_idx']
            if pub_idx:
                sel = random.choice(pub_idx)
                if priv[sel]:
                    cnt_public += 1
            # generate selection
            gen_idx = problem_indices[i]['max_idx_list']
            if gen_idx:
                sel = random.choice(gen_idx)
                if priv[sel]:
                    cnt_generate += 1
        plain_res.append(cnt_plain)
        public_res.append(cnt_public)
        generate_res.append(cnt_generate)
    # stats

    def stats(arr):
        return {
            'min': min(arr), 'max': max(arr), 'mean': np.mean(arr),
            'rate_min': min(arr)/len(df), 'rate_max': max(arr)/len(df), 'rate_mean': np.mean(arr)/len(df)
        }
    st_plain, st_public, st_generate = stats(
        plain_res), stats(public_res), stats(generate_res)
    # upper bound
    ub_count = sum(1 for i in range(len(df))
                   if problem_indices[i]['private_case_true_idx'])
    ub_rate = ub_count/len(df)
    log_message(
        f"{label} upper bound: {ub_count}/{len(df)} = {ub_rate:.4f}", output_file)
    # log selections
    def fmt(
        s): return f"min: {s['min']}, max: {s['max']}, mean: {s['mean']:.1f}, rate min: {s['rate_min']:.4f}, rate max: {s['rate_max']:.4f}, rate mean: {s['rate_mean']:.4f}"
    log_message(
        f"{label}  Plain random      -> {fmt(st_plain)}, ana: {ana_plain:.4f}", output_file)
    log_message(
        f"{label}  Public selection   -> {fmt(st_public)}, ana: {ana_public:.4f}", output_file)
    log_message(
        f"{label}  Generate selection -> {fmt(st_generate)}, ana: {ana_generate:.4f}", output_file)
    log_message(f"----- end {label} -----\n", output_file)


def run_k_analysis(original_df, k, validate_test_cases, n_trials, output_file):
    """
    Perform the same analysis pipeline but only considering the first k trajectories
    (solutions) of each problem. It logs results to `output_file`.
    All existing helper functions are reused; we simply truncate list-valued
    columns to length k before running the pipeline.
    """
    log_message(f"\n===== k = {k} analysis start =====", output_file)

    # 1. Create a shallow copy of the DataFrame structure and slice relevant columns
    df_k = original_df.copy(deep=True)
    for col in ['metadata', 'generated_metadata']:
        if col in df_k.columns:
            df_k[col] = df_k[col].apply(lambda x: _slice_first_k(x, k))

    # 2. Recompute public/private scores for the truncated data
    public_scores_k, private_scores_k = get_public_private_scores(df_k)
    df_k['public_scores'] = public_scores_k
    df_k['private_scores'] = private_scores_k

    # 3. Derive indices & analytical pass rates (public-filtered and unfiltered)
    problem_indices_k_full = get_public_private_generated_right_index(
        df_k, validate_test_cases)

    # Public-filtered generate selection (legacy behavior)
    ana_rates_public = calculate_analytical_pass_rates(
        df_k, problem_indices_k_full)

    # Unfiltered generate selection: copy dicts but remap max_idx_list
    problem_indices_k_all = [
        {
            'public_case_true_idx': d['public_case_true_idx'],
            'private_case_true_idx': d['private_case_true_idx'],
            'max_idx_list': d['max_idx_all']  # use unfiltered best
        } for d in problem_indices_k_full
    ]
    ana_rates_all = calculate_analytical_pass_rates(
        df_k, problem_indices_k_all)

    # 4+5+6+7: simulate, stats, upper bound, log via helper
    _simulate_and_log(df_k, problem_indices_k_full, ana_rates_public,
                      n_trials, output_file, f"k={k} (public-filtered)")
    _simulate_and_log(df_k, problem_indices_k_all, ana_rates_all,
                      n_trials, output_file, f"k={k} (all)")

    # Compute and return upper bound rate along with analytical rates
    ub_count_k = sum(
        1 for info in problem_indices_k_full if info['private_case_true_idx'])
    ub_rate_k = ub_count_k / len(problem_indices_k_full)

    # Return tuple with both generate strategies: (plain, public, generate_public, generate_all)
    return ub_rate_k, (ana_rates_public[0], ana_rates_public[1], ana_rates_public[2], ana_rates_all[2])

    # end run_k_analysis



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_path", type=str,
                        default=None)
    parser.add_argument("--validate_test_cases", action="store_true",
                        help="Whether to enable test case validation")
    args = parser.parse_args()

    response_path = args.response_path
    validate_test_cases = args.validate_test_cases

    # Include validation info in filenames
    validation_suffix = "_validated" if validate_test_cases else "_unvalidated"
    output_path = response_path.replace('.pkl', f'{validation_suffix}.txt')

    response_dataset = pd.read_pickle(response_path)

    if type(response_dataset) == list:
        response_dataset = pd.DataFrame(response_dataset)

    # Open the output file at the beginning
    output_file = open(output_path, 'w')

    # Use DataFrame string representation instead of info()
    log_message(f"DataFrame info:\n{response_dataset.dtypes}\n"
                f"Shape: {response_dataset.shape}", output_file)

    public_scores_list, private_scores_list = get_public_private_scores(
        response_dataset)
    response_dataset['public_scores'] = public_scores_list
    response_dataset['private_scores'] = private_scores_list
    # Use DataFrame string representation instead of info()
    log_message(f"DataFrame info:\n{response_dataset.dtypes}\n"
                f"Shape: {response_dataset.shape}", output_file)

    # Number of random trials
    n_trials = 10

    # --------- k-subset analyses (including full k = original answer number) ---------
    max_k = len(response_dataset.iloc[0]['metadata'])
    ks_to_run = [k for k in (1, 2, 4, 8, 16, max_k) if k <= max_k]
    # Initialize accumulators
    k_values_full = []
    upper_bounds_full = []
    plain_random_full = []
    public_selection_full = []
    generate_selection_public_full = []
    generate_selection_all_full = []

    log_message("\nRunning k-subset analyses for ks=" +
                str(ks_to_run), output_file)
    for k_val in ks_to_run:
        ub_rate, (ana_plain, ana_public, ana_generate_public, ana_generate_all) = run_k_analysis(
            response_dataset, k_val, validate_test_cases, n_trials, output_file)
        k_values_full.append(k_val)
        upper_bounds_full.append(round(ub_rate, 4))
        plain_random_full.append(round(ana_plain, 4))
        public_selection_full.append(round(ana_public, 4))
        generate_selection_public_full.append(round(ana_generate_public, 4))
        generate_selection_all_full.append(round(ana_generate_all, 4))

    # Log summary arrays
    log_message(f"k_values_full = {k_values_full}", output_file)
    log_message(f"upper_bounds_full = {upper_bounds_full}", output_file)
    log_message(f"plain_random_full = {plain_random_full}", output_file)
    log_message(
        f"public_selection_full = {public_selection_full}", output_file)
    log_message(
        f"generate_selection_public_full = {generate_selection_public_full}", output_file)
    log_message(
        f"generate_selection_all_full = {generate_selection_all_full}", output_file)

    # Close the file
    output_file.close()
