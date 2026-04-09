#!/usr/bin/env python3
"""
Test ALL cases of Filesystem MCP edit_file to find where indentation breaks.

Cases tested:
1. Shallow code (4 spaces) - exact match with indent
2. Shallow code (4 spaces) - trimmed oldText (no indent)
3. Deep nested code (12 spaces) - exact match with indent
4. Deep nested code (12 spaces) - trimmed oldText (no indent)
5. Adding NEW lines - with correct absolute indent
6. Adding NEW lines - with relative indent only (what LLM does)
"""

import json
import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mcp.context import MCPClient

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_FILE = os.path.join(_BASE_DIR, "context_runs", "index_workspace", "insertion", "insert_heuristic.py")
TEST_DIR = os.path.join(_BASE_DIR, "test_temp")


def get_client(root_dir: str) -> MCPClient:
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fs_dist = os.path.join(
        pkg_root, "node_modules", "@modelcontextprotocol", "server-filesystem", "dist", "index.js"
    )
    return MCPClient(["node", fs_dist, root_dir], {**os.environ})


def setup_test_file():
    """Copy source to test dir, return path."""
    os.makedirs(TEST_DIR, exist_ok=True)
    test_file = os.path.join(TEST_DIR, "insert_heuristic.py")
    shutil.copy2(SOURCE_FILE, test_file)
    return test_file


def show_lines(filepath, start, end, label=""):
    """Show lines from file with indent info."""
    with open(filepath) as f:
        lines = f.readlines()
    if label:
        print(f"\n{label}:")
    for i in range(start-1, min(end, len(lines))):
        indent = len(lines[i]) - len(lines[i].lstrip())
        print(f"  {i+1}: [{indent:2d} sp] {lines[i].rstrip()[:70]}")


def run_edit(client, filepath, old_text, new_text, case_name):
    """Run an edit and report result."""
    print(f"\n{'='*80}")
    print(f"CASE: {case_name}")
    print(f"{'='*80}")

    print(f"\noldText ({len(old_text.splitlines())} lines):")
    for line in old_text.splitlines():
        indent = len(line) - len(line.lstrip())
        print(f"  [{indent:2d} sp] {repr(line[:50])}")

    print(f"\nnewText ({len(new_text.splitlines())} lines):")
    for line in new_text.splitlines():
        indent = len(line) - len(line.lstrip())
        print(f"  [{indent:2d} sp] {repr(line[:50])}")

    # Dry run
    result = client.call("edit_file", {
        "path": filepath,
        "edits": [{"oldText": old_text, "newText": new_text}],
        "dryRun": True
    }, timeout_s=30.0)

    if result.get("error"):
        print(f"\n❌ FAILED (dry run): {result['error']}")
        return False

    # Apply
    result = client.call("edit_file", {
        "path": filepath,
        "edits": [{"oldText": old_text, "newText": new_text}],
        "dryRun": False
    }, timeout_s=30.0)

    if result.get("error"):
        print(f"\n❌ FAILED (apply): {result['error']}")
        return False

    print("\n✓ Edit applied successfully")
    return True


def test_case_1():
    """Shallow code (4 spaces) - exact match with indent."""
    test_file = setup_test_file()
    client = get_client(TEST_DIR)

    show_lines(test_file, 26, 28, "Before (line 27 has 4-space indent)")

    # Line 27: "    deliveries = set(dropoff_to_pickups.keys())"
    old_text = "    deliveries = set(dropoff_to_pickups.keys())"
    new_text = "    deliveries = set(dropoff_to_pickups.keys())\n    # CASE1: Added with 4-space indent"

    success = run_edit(client, test_file, old_text, new_text, "1: Shallow (4sp) - exact match")

    if success:
        show_lines(test_file, 26, 30, "After")
        # Check indent of new line
        with open(test_file) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "CASE1" in line:
                indent = len(line) - len(line.lstrip())
                print(f"\n→ New line indent: {indent} spaces (expected: 4)")
                print("✓ PASS" if indent == 4 else "❌ FAIL")
                break

    client.close()


def test_case_2():
    """Shallow code (4 spaces) - trimmed oldText (no indent)."""
    test_file = setup_test_file()
    client = get_client(TEST_DIR)

    show_lines(test_file, 26, 28, "Before")

    # oldText WITHOUT leading spaces
    old_text = "deliveries = set(dropoff_to_pickups.keys())"
    new_text = "deliveries = set(dropoff_to_pickups.keys())\n    # CASE2: Added with 4-space indent"

    success = run_edit(client, test_file, old_text, new_text, "2: Shallow (4sp) - trimmed oldText")

    if success:
        show_lines(test_file, 26, 30, "After")
        with open(test_file) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "CASE2" in line:
                indent = len(line) - len(line.lstrip())
                print(f"\n→ New line indent: {indent} spaces (expected: 4)")
                print("✓ PASS" if indent == 4 else "❌ FAIL")
                break

    client.close()


def test_case_3():
    """Deep nested code (12 spaces) - exact match with indent."""
    test_file = setup_test_file()
    client = get_client(TEST_DIR)

    show_lines(test_file, 418, 422, "Before (line 419 has 12-space indent)")

    # Line 419: "            for bus in bus_map.values():"
    old_text = "            for bus in bus_map.values():"
    new_text = "            for bus in bus_map.values():\n                # CASE3: Added with 16-space indent"

    success = run_edit(client, test_file, old_text, new_text, "3: Deep (12sp) - exact match")

    if success:
        show_lines(test_file, 418, 423, "After")
        with open(test_file) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "CASE3" in line:
                indent = len(line) - len(line.lstrip())
                print(f"\n→ New line indent: {indent} spaces (expected: 16)")
                print("✓ PASS" if indent == 16 else "❌ FAIL")
                break

    client.close()


def test_case_4():
    """Deep nested code (12 spaces) - trimmed oldText (no indent)."""
    test_file = setup_test_file()
    client = get_client(TEST_DIR)

    show_lines(test_file, 418, 422, "Before")

    # oldText WITHOUT leading spaces (like LLM often does)
    old_text = "for bus in bus_map.values():"
    new_text = "for bus in bus_map.values():\n    # CASE4: Added with 4-space indent"

    success = run_edit(client, test_file, old_text, new_text, "4: Deep (12sp) - trimmed oldText")

    if success:
        show_lines(test_file, 418, 423, "After")
        with open(test_file) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "CASE4" in line:
                indent = len(line) - len(line.lstrip())
                print(f"\n→ New line indent: {indent} spaces (expected: 16, got via auto-indent?)")
                print("✓ PASS" if indent == 16 else "❌ FAIL - auto-indent did NOT work")
                break

    client.close()


def test_case_5():
    """Adding NEW lines with correct ABSOLUTE indent (what we want LLM to do)."""
    test_file = setup_test_file()
    client = get_client(TEST_DIR)

    show_lines(test_file, 418, 422, "Before")

    # Two-line anchor, inserting constraint between
    old_text = "            for bus in bus_map.values():\n                candidate = evaluate_bus_candidate("
    new_text = """            for bus in bus_map.values():
                if stop == 20006 and bus['id'] != 0:
                    continue
                candidate = evaluate_bus_candidate("""

    success = run_edit(client, test_file, old_text, new_text, "5: Add lines - CORRECT absolute indent")

    if success:
        show_lines(test_file, 418, 425, "After")
        with open(test_file) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "if stop == 20006" in line:
                indent = len(line) - len(line.lstrip())
                print(f"\n→ Constraint line indent: {indent} spaces (expected: 16)")
                print("✓ PASS" if indent == 16 else "❌ FAIL")
                break

    client.close()


def test_case_6():
    """Adding NEW lines with relative indent only (what LLM actually does)."""
    test_file = setup_test_file()
    client = get_client(TEST_DIR)

    show_lines(test_file, 418, 422, "Before")

    # Trimmed anchor, relative 4-space indent (like LLM does)
    old_text = "for bus in bus_map.values():\n    candidate = evaluate_bus_candidate("
    new_text = """for bus in bus_map.values():
    if stop == 20006 and bus['id'] != 0:
        continue
    candidate = evaluate_bus_candidate("""

    success = run_edit(client, test_file, old_text, new_text, "6: Add lines - RELATIVE indent (LLM style)")

    if success:
        show_lines(test_file, 418, 425, "After")
        with open(test_file) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "if stop == 20006" in line:
                indent = len(line) - len(line.lstrip())
                print(f"\n→ Constraint line indent: {indent} spaces (expected: 16)")
                print("✓ PASS" if indent == 16 else f"❌ FAIL - got {indent} spaces (auto-indent broken)")
                break

    client.close()


def fix_indentation(file_path: str, old_text: str, new_text: str) -> tuple:
    """Add file's base indentation to all lines. Expects PURE RELATIVE indent."""
    with open(file_path, 'r') as f:
        file_content = f.read()

    anchor = old_text.split('\n')[0].strip()

    # Find anchor line in file
    file_indent = ""
    for line in file_content.split('\n'):
        if line.strip() == anchor:
            file_indent = line[:len(line) - len(line.lstrip())]
            break

    if not file_indent:
        return old_text, new_text

    # Check if already has file's indentation
    first_line = old_text.split('\n')[0]
    if len(first_line) - len(first_line.lstrip()) == len(file_indent):
        return old_text, new_text

    # Add file_indent to all non-empty lines
    def add_indent(text):
        result = []
        for line in text.split('\n'):
            if line.strip():
                result.append(file_indent + line)
            else:
                result.append(line)
        return '\n'.join(result)

    return add_indent(old_text), add_indent(new_text)


def test_case_7():
    """Case 6 + helper fix: LLM style relative indent -> fixed to absolute."""
    test_file = setup_test_file()
    client = get_client(TEST_DIR)

    show_lines(test_file, 418, 422, "Before")

    # Same as Case 6: trimmed anchor, relative 4-space indent (LLM style)
    old_text = "for bus in bus_map.values():\n    candidate = evaluate_bus_candidate("
    new_text = """for bus in bus_map.values():
    if stop == 20006 and bus['id'] != 0:
        continue
    candidate = evaluate_bus_candidate("""

    print("\n--- Before fix ---")
    print(f"old_text first line indent: {len(old_text.split(chr(10))[0]) - len(old_text.split(chr(10))[0].lstrip())}")

    fixed_old, fixed_new = fix_indentation(test_file, old_text, new_text)

    print(f"\n--- After fix ---")
    print(f"fixed_old first line indent: {len(fixed_old.split(chr(10))[0]) - len(fixed_old.split(chr(10))[0].lstrip())}")

    success = run_edit(client, test_file, fixed_old, fixed_new, "7: Case 6 + HELPER FIX")

    if success:
        show_lines(test_file, 418, 425, "After")
        with open(test_file) as f:
            lines = f.readlines()
        for line in lines:
            if "if stop == 20006" in line:
                indent = len(line) - len(line.lstrip())
                print(f"\n→ Constraint line indent: {indent} spaces (expected: 16)")
                print("✓ PASS - Helper fix works!" if indent == 16 else f"❌ FAIL - got {indent}")
                break

    client.close()


def test_case_8():
    """Case 5 through helper - already correct indent should NOT be adjusted."""
    test_file = setup_test_file()
    client = get_client(TEST_DIR)

    show_lines(test_file, 418, 422, "Before")

    # Same as Case 5: CORRECT absolute indent
    old_text = "            for bus in bus_map.values():\n                candidate = evaluate_bus_candidate("
    new_text = """            for bus in bus_map.values():
                if stop == 20006 and bus['id'] != 0:
                    continue
                candidate = evaluate_bus_candidate("""

    print("\n--- Testing helper with already-correct indent ---")
    fixed_old, fixed_new = fix_indentation(test_file, old_text, new_text)

    # Verify helper didn't change anything
    if fixed_old == old_text and fixed_new == new_text:
        print("✓ Helper correctly detected already-correct indent (no change)")
    else:
        print("❌ Helper incorrectly modified already-correct indent!")

    success = run_edit(client, test_file, fixed_old, fixed_new, "8: Already correct indent + helper (should not adjust)")

    if success:
        show_lines(test_file, 418, 425, "After")
        with open(test_file) as f:
            lines = f.readlines()
        for line in lines:
            if "if stop == 20006" in line:
                indent = len(line) - len(line.lstrip())
                print(f"\n→ Constraint line indent: {indent} spaces (expected: 16)")
                print("✓ PASS" if indent == 16 else f"❌ FAIL - got {indent}")
                break

    client.close()


def test_case_9():
    """Mixed indentation (line 0 at col 0, others have 12sp) - normalize then add file_indent."""
    test_file = setup_test_file()
    client = get_client(TEST_DIR)

    show_lines(test_file, 406, 410, "Before")

    # This is what the LLM actually produced - mixed indentation
    # Line 0: no indent, Line 1+: has 12-space indent
    old_text = "best_metrics = None\n            "
    new_text = "best_metrics = None\n            if stop == 20006:\n                best_bus_id = 0\n                break\n            "

    print("\n--- Mixed indentation test (LLM style) ---")
    print(f"old_text lines:")
    for i, line in enumerate(old_text.split('\n')):
        print(f"  {i}: [{len(line) - len(line.lstrip()):2d} sp] {repr(line[:40])}")
    print(f"new_text lines:")
    for i, line in enumerate(new_text.split('\n')):
        print(f"  {i}: [{len(line) - len(line.lstrip()):2d} sp] {repr(line[:40])}")

    fixed_old, fixed_new = fix_indentation(test_file, old_text, new_text)

    print(f"\n--- After fix ---")
    print(f"fixed_old lines:")
    for i, line in enumerate(fixed_old.split('\n')):
        print(f"  {i}: [{len(line) - len(line.lstrip()):2d} sp] {repr(line[:40])}")
    print(f"fixed_new lines:")
    for i, line in enumerate(fixed_new.split('\n')):
        print(f"  {i}: [{len(line) - len(line.lstrip()):2d} sp] {repr(line[:40])}")

    success = run_edit(client, test_file, fixed_old, fixed_new, "9: Mixed indent (normalize + add file_indent)")

    if success:
        show_lines(test_file, 406, 414, "After")
        with open(test_file) as f:
            lines = f.readlines()
        for line in lines:
            if "if stop == 20006" in line:
                indent = len(line) - len(line.lstrip())
                print(f"\n→ if-statement indent: {indent} spaces (expected: 12)")
                print("✓ PASS" if indent == 12 else f"❌ FAIL - got {indent}")
                break

    client.close()


def test_case_10():
    """NO HELPER: Single anchor line + relative indent for new lines. Can MCP auto-indent?"""
    test_file = setup_test_file()
    client = get_client(TEST_DIR)

    show_lines(test_file, 406, 412, "Before (best_metrics at 12sp)")

    # Single-line anchor (no indent in edit), new lines with relative indent
    # LLM provides: anchor at col 0, new lines at 4sp (one level deeper)
    old_text = "best_metrics = None"
    new_text = """best_metrics = None
    if stop == 20006:
        best_bus_id = 0
        continue"""

    print("\n--- NO HELPER: Single anchor + relative indent ---")
    print(f"oldText: {repr(old_text)}")
    print(f"newText lines:")
    for i, line in enumerate(new_text.split('\n')):
        print(f"  {i}: [{len(line) - len(line.lstrip()):2d} sp] {repr(line)}")

    # NO helper - direct to MCP
    success = run_edit(client, test_file, old_text, new_text, "10: NO HELPER - single anchor + relative indent")

    if success:
        show_lines(test_file, 406, 414, "After")
        with open(test_file) as f:
            lines = f.readlines()
        for line in lines:
            if "if stop == 20006" in line:
                indent = len(line) - len(line.lstrip())
                print(f"\n→ if-statement indent: {indent} spaces (expected: 16 = 12 base + 4 relative)")
                print("✓ PASS" if indent == 16 else f"❌ FAIL - got {indent}")
                break

    client.close()


def test_case_11():
    """NO HELPER: Replace follower line with new code + follower. MCP auto-indent on line 0."""
    test_file = setup_test_file()
    client = get_client(TEST_DIR)

    show_lines(test_file, 418, 424, "Before (candidate at 16sp)")

    # Replace the follower line (candidate) with: new code + follower
    # This way line 0 of newText gets auto-indented to 16 spaces (same as candidate)
    old_text = "candidate = evaluate_bus_candidate("
    new_text = """if stop == 20006 and bus['id'] != 0:
    continue
candidate = evaluate_bus_candidate("""

    print("\n--- NO HELPER: Replace follower with new+follower ---")
    print(f"oldText: {repr(old_text)}")
    print(f"newText lines:")
    for i, line in enumerate(new_text.split('\n')):
        print(f"  {i}: [{len(line) - len(line.lstrip()):2d} sp] {repr(line[:50])}")

    # NO helper - direct to MCP
    success = run_edit(client, test_file, old_text, new_text, "11: NO HELPER - replace follower with new+follower")

    if success:
        show_lines(test_file, 418, 426, "After")
        with open(test_file) as f:
            lines = f.readlines()
        for line in lines:
            if "if stop == 20006" in line:
                indent = len(line) - len(line.lstrip())
                print(f"\n→ if-statement indent: {indent} spaces (expected: 16)")
                print("✓ PASS" if indent == 16 else f"❌ FAIL - got {indent}")
                break

    client.close()


def test_case_12():
    """MULTI-EDIT: Each new line as line 0 of separate edit - MCP auto-indents each."""
    test_file = setup_test_file()
    client = get_client(TEST_DIR)

    show_lines(test_file, 418, 424, "Before")

    print("\n--- MULTI-EDIT approach: 2 separate edits ---")

    # Edit 1: Insert if-statement before candidate (if becomes line 0)
    old1 = "candidate = evaluate_bus_candidate("
    new1 = "if stop == 20006 and bus['id'] != 0:\ncandidate = evaluate_bus_candidate("

    print(f"\nEdit 1 - Insert if before candidate:")
    print(f"  old: {repr(old1)}")
    print(f"  new: {repr(new1[:60])}...")

    result1 = client.call("edit_file", {
        "path": test_file,
        "edits": [{"oldText": old1, "newText": new1}],
        "dryRun": False
    }, timeout_s=30.0)

    if result1.get("error"):
        print(f"  ❌ Edit 1 failed: {result1['error']}")
        client.close()
        return

    print("  ✓ Edit 1 applied")
    show_lines(test_file, 418, 424, "After Edit 1")

    # Edit 2: Insert continue inside the if (continue becomes line 0)
    old2 = "if stop == 20006 and bus['id'] != 0:"
    new2 = "if stop == 20006 and bus['id'] != 0:\n    continue"

    print(f"\nEdit 2 - Insert continue inside if:")
    print(f"  old: {repr(old2)}")
    print(f"  new: {repr(new2)}")

    result2 = client.call("edit_file", {
        "path": test_file,
        "edits": [{"oldText": old2, "newText": new2}],
        "dryRun": False
    }, timeout_s=30.0)

    if result2.get("error"):
        print(f"  ❌ Edit 2 failed: {result2['error']}")
        client.close()
        return

    print("  ✓ Edit 2 applied")
    show_lines(test_file, 418, 426, "After Edit 2 (Final)")

    # Verify all lines
    with open(test_file) as f:
        lines = f.readlines()

    print("\n--- Verification ---")
    for line in lines:
        if "if stop == 20006" in line:
            indent = len(line) - len(line.lstrip())
            print(f"if-statement: {indent} spaces (expected: 16)")
            if indent != 16:
                print("❌ FAIL")
                break
        if "continue" in line and "continue" == line.strip():
            indent = len(line) - len(line.lstrip())
            print(f"continue: {indent} spaces (expected: 20)")
            if indent != 20:
                print("❌ FAIL")
                break
        if "candidate = evaluate_bus_candidate" in line:
            indent = len(line) - len(line.lstrip())
            print(f"candidate: {indent} spaces (expected: 16)")
            if indent == 16:
                print("✓ ALL PASS - Multi-edit approach works!")
            else:
                print("❌ FAIL")
            break

    client.close()


def test_case_13():
    """PURE RELATIVE indent + helper: What LLM should produce."""
    test_file = setup_test_file()
    client = get_client(TEST_DIR)

    show_lines(test_file, 418, 424, "Before (for at 12sp, candidate at 16sp)")

    # PURE RELATIVE indentation - anchor at col 0, body at 4sp relative
    # This is what the updated system prompt tells LLM to produce
    old_text = "for bus in bus_map.values():"
    new_text = """for bus in bus_map.values():
    if stop == 20006 and bus['id'] != 0:
        continue"""

    print("\n--- PURE RELATIVE indent (what LLM should produce) ---")
    print(f"oldText: {repr(old_text)}")
    print(f"newText lines (all relative to col 0):")
    for i, line in enumerate(new_text.split('\n')):
        print(f"  {i}: [{len(line) - len(line.lstrip()):2d} sp] {repr(line)}")

    # Apply helper
    fixed_old, fixed_new = fix_indentation(test_file, old_text, new_text)

    print(f"\nAfter helper adds file indent (12sp):")
    for i, line in enumerate(fixed_new.split('\n')):
        print(f"  {i}: [{len(line) - len(line.lstrip()):2d} sp] {repr(line[:50])}")

    success = run_edit(client, test_file, fixed_old, fixed_new, "13: PURE RELATIVE + helper")

    if success:
        show_lines(test_file, 418, 424, "After")
        with open(test_file) as f:
            lines = f.readlines()

        all_pass = True
        for line in lines:
            if "for bus in bus_map" in line:
                indent = len(line) - len(line.lstrip())
                print(f"for loop: {indent} spaces (expected: 12)")
                if indent != 12:
                    all_pass = False
            if "if stop == 20006" in line:
                indent = len(line) - len(line.lstrip())
                print(f"if statement: {indent} spaces (expected: 16)")
                if indent != 16:
                    all_pass = False
            if line.strip() == "continue":
                indent = len(line) - len(line.lstrip())
                print(f"continue: {indent} spaces (expected: 20)")
                if indent != 20:
                    all_pass = False

        if all_pass:
            print("✓ ALL PASS - Pure relative + helper works!")
        else:
            print("❌ FAIL")

    client.close()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("FILESYSTEM MCP EDIT_FILE - COMPREHENSIVE INDENT TESTS")
    print("="*80)

    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()
    test_case_6()
    # Skip helper tests for now
    # test_case_7()
    # test_case_8()
    # test_case_9()
    # test_case_10()  # NO HELPER tests
    # test_case_11()
    # test_case_12()  # MULTI-EDIT approach
    test_case_13()  # PURE RELATIVE indent + helper

    # Cleanup
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
        print(f"\nCleaned up {TEST_DIR}")

    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
