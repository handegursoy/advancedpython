"""
Programming Assignment 1
-----------------------
Author: Hande Gürsoy
Date: 08/11/2024

This notebook contains solutions to five programming problems focusing on basic Python concepts
including loops, conditionals, string manipulation, and algorithmic thinking.
"""

import random
import time  # For adding slight delays in the game for better user experience

# Problem 1: Number Guessing Game
def number_guessing_game():
    """
    Implementation of a number guessing game where the user needs to guess
    a random number between 1 and 20.
    
    Features:
    - Random number generation
    - Input validation
    - Informative feedback
    - Attempt counter
    """
    correct_number = random.randint(1, 20)
    attempts = 0
    
    print("\n=== Welcome to the Number Guessing Game! ===")
    print("Try to guess the number between 1 and 20.")
    print("I'll tell you if your guess is too high or too low.\n")
    
    while True:
        try:
            guess = int(input("Enter your guess: "))
            attempts += 1
            
            # Input validation
            if guess < 1 or guess > 20:
                print("Please enter a number between 1 and 20!")
                attempts -= 1  # Don't count invalid attempts
                continue
            
            # Check guess
            if guess < correct_number:
                print("Too low! Try a higher number.")
            elif guess > correct_number:
                print("Too high! Try a lower number.")
            else:
                print(f"\nCongratulations! You've guessed the number {correct_number} correctly!")
                print(f"It took you {attempts} attempts.")
                break
                
        except ValueError:
            print("Please enter a valid number!")
            attempts -= 1  # Don't count invalid attempts


# Problem 2: Search Insert Position
def search_insert(nums, target):
    """
    Finds the index where target is found or should be inserted in a sorted array.
    
    Args:
        nums (List[int]): Sorted array of distinct integers
        target (int): Number to find or insert
    
    Returns:
        int: Index where target is found or should be inserted
    
    Time Complexity: O(log n) - Binary Search
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return left


# Problem 3: Binary String Addition
def add_binary(a, b):
    """
    Adds two binary strings and returns their sum as a binary string.
    
    Args:
        a (str): First binary string
        b (str): Second binary string
    
    Returns:
        str: Sum of the binary strings
    
    Example:
        >>> add_binary("11", "1")
        "100"
    """
    # Convert binary strings to integers
    num_a = int(a, 2)
    num_b = int(b, 2)
    
    # Add numbers
    sum_decimal = num_a + num_b
    
    # Convert back to binary and remove '0b' prefix
    return bin(sum_decimal)[2:]


# Problem 4: Find Single Number
def find_single_number(nums):
    """
    Finds the number that appears only once in an array where all other
    numbers appear exactly twice.
    
    Args:
        nums (List[int]): Array of integers
    
    Returns:
        int: The number that appears only once
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    result = 0
    for num in nums:
        # Using XOR operation: n^n = 0, and n^0 = n
        # Numbers that appear twice will cancel out
        result ^= num
    return result


# Problem 5: Repeated DNA Sequences
def find_repeated_sequences(s):
    """
    Finds all 10-letter sequences that occur more than once in a DNA string.
    
    Args:
        s (str): DNA sequence string containing only A, C, G, and T
    
    Returns:
        List[str]: List of all repeated 10-letter sequences
    
    Time Complexity: O(n), where n is the length of the string
    Space Complexity: O(n)
    """
    # Validate input
    if len(s) < 10:
        return []
    
    seen = set()
    repeated = set()
    
    # Check all possible 10-letter sequences
    for i in range(len(s) - 9):
        curr_seq = s[i:i+10]
        if curr_seq in seen:
            repeated.add(curr_seq)
        else:
            seen.add(curr_seq)
            
    return list(repeated)


# Test Functions
def run_tests():
    """
    Comprehensive test suite for all implemented functions.
    Tests various cases including edge cases and typical scenarios.
    """
    print("\n=== Running Comprehensive Tests ===\n")
    
    # Test Problem 2: Search Insert Position
    print("Testing Search Insert Position:")
    test_cases = [
        ([1, 3, 5, 6], 5, 2),  # Normal case - number exists
        ([1, 3, 5, 6], 2, 1),  # Insert in middle
        ([1, 3, 5, 6], 7, 4),  # Insert at end
        ([1, 3, 5, 6], 0, 0),  # Insert at beginning
        ([], 1, 0)             # Empty array
    ]
    
    for nums, target, expected in test_cases:
        result = search_insert(nums, target)
        print(f"nums={nums}, target={target}")
        print(f"Expected: {expected}, Got: {result}")
        assert result == expected, f"Test failed: Expected {expected}, got {result}"
    
    # Test Problem 3: Binary Addition
    print("\nTesting Binary Addition:")
    test_cases = [
        ("11", "1", "100"),
        ("1010", "1011", "10101"),
        ("0", "0", "0"),
        ("1", "111", "1000")
    ]
    
    for a, b, expected in test_cases:
        result = add_binary(a, b)
        print(f"a={a}, b={b}")
        print(f"Expected: {expected}, Got: {result}")
        assert result == expected, f"Test failed: Expected {expected}, got {result}"
    
    # Test Problem 4: Single Number
    print("\nTesting Single Number:")
    test_cases = [
        ([2,2,1], 1),
        ([4,1,2,1,2], 4),
        ([1], 1)
    ]
    
    for nums, expected in test_cases:
        result = find_single_number(nums)
        print(f"nums={nums}")
        print(f"Expected: {expected}, Got: {result}")
        assert result == expected, f"Test failed: Expected {expected}, got {result}"
    
    # Test Problem 5: DNA Sequences
    print("\nTesting DNA Sequences:")
    test_cases = [
        ("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT", ["AAAAACCCCC", "CCCCCAAAAA"]),
        ("AAAAAAAAAA", []),
        ("AAAAAAAAAAA", ["AAAAAAAAAA"])
    ]
    
    for s, expected in test_cases:
        result = sorted(find_repeated_sequences(s))
        expected = sorted(expected)
        print(f"Input: {s}")
        print(f"Expected: {expected}, Got: {result}")
        assert result == expected, f"Test failed: Expected {expected}, got {result}"
    
    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    # Run test suite
    run_tests()
    
    # Uncomment to play the number guessing game
    # number_guessing_game()
