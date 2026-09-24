TEST_CASE_TYPES = {
    "basic": "basic test case that validates core functionality with simple, straightforward inputs",
    "edge": "edge case that tests boundary values and constraint limits (minimum/maximum allowed values)",
    "corner": "corner case with unusual inputs like empty collections, single elements, or patterns that might break naive solutions",
    "performance": "performance test with large inputs approaching the problem's limits to evaluate solution efficiency"
}

TEST_GENERATION_PROMPT = (
    "You are an expert TEST CASE GENERATOR for programming competitions. Your ONLY task is to generate ONE TEST CASE of a specific type, NOT to solve the problem or write any implementation code.\n\n"
    "Follow these strict guidelines:\n"
    "1. DO NOT write any solution code in any programming language.\n"
    "2. DO NOT attempt to fix or improve the solution.\n"
    "3. FOCUS EXCLUSIVELY on generating a {test_case_type} that can reveal flaws or confirm correctness.\n"
    "4. The test case you generate MUST NOT be identical to any Example Test Case provided in the problem statement.\n\n"
    "You have been provided with:\n"
    "* Problem Description: \n{problem}\n"
    "* A SOLUTION CODE THAT MAY CONTAIN LOGIC ERRORS:\n"
    "```python\n{code}\n```\n\n"
    "Your goal is to create a test case that is valid under the problem constraints and is likely to expose incorrect behavior in the provided code if such errors exist.\n\n"
    "For the test case you create, provide:\n"
    "* The test input exactly as it would be fed to a program.\n"
    "* The expected output that a correct solution should produce.\n"
    "* A brief explanation of what aspect this test case is verifying or how it could reveal flaws.\n\n"
    "Ensure the output follows the *Expected Output Format* structure provided. You must enclose the output in a ```json``` block to facilitate easy extraction and processing.\n\n"
    "Expected Output Format: \n"
    "```json\n"
    "{expected_output}\n"
    "```\n\n"
    "You must ONLY provide ONE {test_case_type}. Ensure your test case is valid according to the problem constraints."
)
