#!/usr/bin/env python3
"""
Membrane Builder

Edit args.txt, then run:
    python run.py

Or specify a different args file:
    python run.py my_args.txt
"""

import os
import sys
import re

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from membrane_builder import MembraneBuilder, MembraneConfig, LipidLibrary
from membrane_builder.validate import MembraneValidator


def parse_args_file(filepath):
    """Parse the args.txt configuration file."""
    config = {
        'box_x': 80.0,
        'box_y': 80.0,
        'box_z': 120.0,
        'bending_modulus': None,
        'temperature': 310.15,
        'min_anchor_distance': 6.5,
        'seed': 12345,
        'output_dir': 'Outputs',
        'output_name': 'membrane',
    }
    lipids = {}

    library = LipidLibrary()
    available_lipids = set(library.list_all())

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.split('#')[0].strip()
            if not line or '=' not in line:
                continue

            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            if not value:
                continue

            if key.upper() in available_lipids:
                counts = re.split(r'[,\s]+', value)
                if len(counts) >= 2:
                    try:
                        top = int(counts[0])
                        bottom = int(counts[1])
                        lipids[key.upper()] = (top, bottom)
                    except ValueError:
                        print(f"Warning: Invalid lipid counts on line {line_num}: {line}")
            else:
                key_lower = key.lower().replace(' ', '_')

                if key_lower in ['box_x', 'box_y', 'box_z', 'temperature', 'min_anchor_distance']:
                    try:
                        config[key_lower] = float(value)
                    except ValueError:
                        print(f"Warning: Invalid number on line {line_num}: {line}")

                elif key_lower == 'bending_modulus':
                    try:
                        config['bending_modulus'] = float(value)
                    except ValueError:
                        config['bending_modulus'] = None

                elif key_lower == 'seed':
                    try:
                        config['seed'] = int(value)
                    except ValueError:
                        pass

                elif key_lower in ['output_dir', 'output_name']:
                    config[key_lower] = value

    config['lipids'] = lipids
    return config


def main():
    print()
    print("MEMBRANE BUILDER")
    print("-" * 40)

    # Get args file
    if len(sys.argv) > 1:
        args_file = sys.argv[1]
    else:
        args_file = os.path.join(os.path.dirname(__file__), 'args.txt')

    if not os.path.exists(args_file):
        print(f"Error: Args file not found: {args_file}")
        print()
        print("Create an args.txt file or specify one:")
        print("  python run.py my_args.txt")
        return 1

    print(f"Config: {args_file}")
    config = parse_args_file(args_file)

    if not config['lipids']:
        print()
        print("Error: No lipids specified.")
        print("Edit args.txt and add lipid lines like:")
        print("  POPC = 64, 64")
        return 1

    # Display configuration
    print(f"Box: {config['box_x']} x {config['box_y']} x {config['box_z']} A")
    print()
    print("Composition:")
    total_top = 0
    total_bottom = 0
    for name, (top, bottom) in config['lipids'].items():
        print(f"  {name}: {top} top, {bottom} bottom")
        total_top += top
        total_bottom += bottom
    print(f"  Total: {total_top + total_bottom} lipids")

    if config['bending_modulus']:
        print(f"Bending modulus: {config['bending_modulus']} kT (user-specified)")
    else:
        print("Bending modulus: calculated from composition")

    # Run validation
    print()
    print("Validating...")
    validator = MembraneValidator()
    validator.validate(
        lipids=config['lipids'],
        box_x=config['box_x'],
        box_y=config['box_y'],
        min_anchor_dist=config['min_anchor_distance'],
    )

    validation_report = validator.print_report()

    if validator.has_errors():
        print(validation_report)
        print()
        print("Build stopped. Fix errors above and try again.")
        return 1

    if validator.has_warnings():
        print(validation_report)
    else:
        print("  OK")

    # Build membrane
    print()
    print("Building...")

    membrane_config = MembraneConfig.create_simple(
        lipids=config['lipids'],
        box_size=(config['box_x'], config['box_y'], config['box_z']),
        bending_modulus=config['bending_modulus'],
    )
    membrane_config.packing.min_anchor_dist = config['min_anchor_distance']

    builder = MembraneBuilder(seed=config['seed'])

    try:
        membrane = builder.build(membrane_config)
    except RuntimeError as e:
        print(f"Error: {e}")
        print()
        print("Try increasing box size or reducing lipid count.")
        return 1

    # Save output files
    os.makedirs(config['output_dir'], exist_ok=True)
    pdb_path = os.path.join(config['output_dir'], config['output_name'] + '.pdb')
    gro_path = os.path.join(config['output_dir'], config['output_name'] + '.gro')
    report_path = os.path.join(config['output_dir'], config['output_name'] + '_report.txt')

    membrane.write_pdb(pdb_path)
    membrane.write_gro(gro_path)

    # Write report
    with open(report_path, 'w') as f:
        f.write("MEMBRANE BUILDER REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Box: {config['box_x']} x {config['box_y']} x {config['box_z']} A\n\n")
        f.write("Composition:\n")
        for name, (top, bottom) in config['lipids'].items():
            f.write(f"  {name}: {top} top, {bottom} bottom\n")
        f.write(f"\nTotal lipids: {membrane.n_lipids}\n")
        f.write(f"Total atoms: {membrane.n_atoms}\n")
        if membrane.properties:
            props = membrane.properties
            f.write(f"\nPhysical Properties:\n")
            f.write(f"  Bending modulus:  {props.bending_modulus:.2f} kT\n")
            f.write(f"  Thickness:        {props.thickness:.1f} A\n")
            f.write(f"  Area per lipid:   {props.area_per_lipid:.1f} A^2\n")
            f.write(f"  Net charge:       {props.net_charge:+.0f} e\n")

    # Display results
    print()
    print("Results:")
    if membrane.properties:
        props = membrane.properties
        print(f"  Bending modulus:  {props.bending_modulus:.2f} kT")
        print(f"  Thickness:        {props.thickness:.1f} A")
        print(f"  Area per lipid:   {props.area_per_lipid:.1f} A^2")
        print(f"  Net charge:       {props.net_charge:+.0f} e")
    print(f"  Lipids:           {membrane.n_lipids}")
    print(f"  Atoms:            {membrane.n_atoms}")

    print()
    print("Output files:")
    print(f"  {pdb_path}")
    print(f"  {gro_path}")
    print(f"  {report_path}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
