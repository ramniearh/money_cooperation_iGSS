"""
Tree Visualization Tool for iGSS

Side-car diagnostic tool for rendering GP trees as pretty diagrams.
This is completely separate from the core code - run it after experiments finish.

Usage:
    python tools/tree_drawer.py --input mode1_experiment_20240101_120000.json
    python tools/tree_drawer.py --string "add(ARG0, mul(ARG1, 2))"

Requirements:
    pip install graphviz
"""

import argparse
import json
import re
from pathlib import Path


def parse_tree_string(tree_str):
    """
    Parse a DEAP tree string into a nested structure.
    
    Example:
        "add(ARG0, mul(ARG1, 2))" -> 
        {'name': 'add', 'children': [
            {'name': 'ARG0'},
            {'name': 'mul', 'children': [
                {'name': 'ARG1'},
                {'name': '2'}
            ]}
        ]}
    """
    # Remove whitespace
    tree_str = tree_str.replace(' ', '')
    
    def parse_node(s, pos=0):
        """Recursively parse a node starting at position pos."""
        # Find the node name (until '(' or ',' or ')')
        name_end = pos
        while name_end < len(s) and s[name_end] not in '(),':
            name_end += 1
        
        name = s[pos:name_end]
        
        # If no opening paren, this is a leaf node
        if name_end >= len(s) or s[name_end] != '(':
            return {'name': name}, name_end
        
        # This is a parent node - parse children
        children = []
        pos = name_end + 1  # Skip '('
        
        while pos < len(s) and s[pos] != ')':
            child, pos = parse_node(s, pos)
            children.append(child)
            
            # Skip comma if present
            if pos < len(s) and s[pos] == ',':
                pos += 1
        
        # Skip ')'
        if pos < len(s) and s[pos] == ')':
            pos += 1
        
        return {'name': name, 'children': children}, pos
    
    tree, _ = parse_node(tree_str)
    return tree


def tree_to_dot(tree, node_id=0):
    """
    Convert parsed tree to Graphviz DOT format.
    
    Returns:
        tuple: (dot_string, next_node_id)
    """
    name = tree['name']
    current_id = node_id
    
    # Color nodes by type
    if name.startswith('ARG'):
        color = 'lightgreen'  # Inputs
        shape = 'ellipse'
    elif name in ['add', 'sub', 'mul', 'if_then']:
        color = 'lightblue'  # Operators
        shape = 'box'
    else:
        color = 'lightyellow'  # Constants
        shape = 'diamond'
    
    dot = f'    node{current_id} [label="{name}", fillcolor={color}, shape={shape}, style=filled];\n'
    next_id = current_id + 1
    
    if 'children' in tree:
        for child in tree['children']:
            child_dot, next_id = tree_to_dot(child, next_id)
            dot += child_dot
            dot += f'    node{current_id} -> node{next_id - len(child.get("children", [])) - 1 if "children" in child else next_id - 1};\n'
    
    return dot, next_id


def render_tree(tree_str, output_file='tree', format='png'):
    """
    Render a tree string to an image file.
    
    Args:
        tree_str: DEAP tree string (e.g., "add(ARG0, mul(ARG1, 2))")
        output_file: Output filename (without extension)
        format: Output format (png, svg, pdf, etc.)
    
    Returns:
        str: Path to output file
    """
    try:
        from graphviz import Source
    except ImportError:
        print("Error: graphviz not installed. Run: pip install graphviz")
        return None
    
    # Parse tree
    tree = parse_tree_string(tree_str)
    
    # Generate DOT
    dot_body, _ = tree_to_dot(tree)
    dot = f"digraph Tree {{\n{dot_body}}}"
    
    # Render
    src = Source(dot)
    output_path = src.render(output_file, format=format, cleanup=True)
    print(f"Tree rendered to: {output_path}")
    return output_path


def extract_tree_from_json(json_file, tree_key='best_strategy'):
    """
    Extract tree string from experiment JSON file.
    
    Args:
        json_file: Path to JSON file
        tree_key: Key to extract (best_strategy, best_assessment_rule, etc.)
    
    Returns:
        str: Tree string or None
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Mode 1 or 2: single tree
    if tree_key in data:
        return data[tree_key]
    
    # Mode 3: dual trees
    if 'best_action_rule' in data and 'best_assessment_rule' in data:
        return {
            'action': data['best_action_rule'],
            'assessment': data['best_assessment_rule']
        }
    
    # Try fossil record
    if 'history' in data and 'fossil_record' in data['history']:
        fossil = data['history']['fossil_record']
        last_gen = max(fossil.keys())
        return fossil[str(last_gen)]
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Visualize GP trees from iGSS experiments'
    )
    parser.add_argument(
        '--input', '-i',
        help='Path to experiment JSON file'
    )
    parser.add_argument(
        '--string', '-s',
        help='Direct tree string to render'
    )
    parser.add_argument(
        '--output', '-o',
        default='tree',
        help='Output filename (without extension)'
    )
    parser.add_argument(
        '--format', '-f',
        default='png',
        choices=['png', 'svg', 'pdf', 'jpg'],
        help='Output format'
    )
    parser.add_argument(
        '--mode', '-m',
        type=int,
        choices=[1, 2, 3],
        help='Mode number (for parsing dual trees in Mode 3)'
    )
    
    args = parser.parse_args()
    
    if args.string:
        # Render direct string
        render_tree(args.string, args.output, args.format)
    
    elif args.input:
        # Extract from JSON
        tree_data = extract_tree_from_json(args.input)
        
        if tree_data is None:
            print(f"Error: Could not extract tree from {args.input}")
            return
        
        # Handle dual trees (Mode 3)
        if isinstance(tree_data, dict):
            print("Rendering dual trees from Mode 3...")
            
            # Parse the combined format: "ACT: ... | ASSESS: ..."
            action_str = tree_data.get('action', '')
            assess_str = tree_data.get('assessment', '')
            
            if action_str:
                render_tree(action_str, f"{args.output}_action", args.format)
            if assess_str:
                render_tree(assess_str, f"{args.output}_assessment", args.format)
        else:
            # Single tree
            render_tree(tree_data, args.output, args.format)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
