#!/usr/bin/env python3
"""
Membrane Builder Command Line Interface
=======================================

A command-line tool for building lipid bilayer membranes.

Usage:
    membrane-builder build --config config.yaml
    membrane-builder quick --lipids POPC:64:64 CHOL:32:32 --box 80 80 120
    membrane-builder list-lipids
    membrane-builder info POPC
    membrane-builder calculate --lipids POPC:64 POPE:32 CHOL:32
"""

import argparse
import sys
import os
from pathlib import Path

from .lipids import LipidLibrary, LipidCategory
from .builder import MembraneBuilder
from .config import MembraneConfig
from .physics import MembranePhysics


def cmd_build(args):
    """Build membrane from configuration file."""
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return 1

    print(f"Loading configuration from: {args.config}")
    config = MembraneConfig.from_yaml(args.config)

    # Validate
    messages = config.validate()
    for msg in messages:
        print(msg)
        if msg.startswith("ERROR"):
            return 1

    print("\n" + config.summary())

    # Build
    print("\nBuilding membrane...")
    builder = MembraneBuilder(seed=args.seed)
    membrane = builder.build(
        config,
        use_templates=args.use_templates,
        templates_dir=args.templates_dir,
    )

    # Output
    os.makedirs(config.output.out_dir, exist_ok=True)

    if config.output.write_pdb:
        pdb_path = os.path.join(config.output.out_dir, "membrane.pdb")
        membrane.write_pdb(pdb_path)
        print(f"Wrote: {pdb_path}")

    if config.output.write_gro:
        gro_path = os.path.join(config.output.out_dir, "membrane.gro")
        membrane.write_gro(gro_path)
        print(f"Wrote: {gro_path}")

    # Properties
    if membrane.properties:
        print("\n" + membrane.properties.summary())

    print(f"\nTotal atoms: {membrane.n_atoms}")
    print(f"Total lipids: {membrane.n_lipids}")

    return 0


def cmd_quick(args):
    """Quick membrane build from command line arguments."""
    # Parse lipids
    lipids = {}
    for spec in args.lipids:
        parts = spec.split(":")
        if len(parts) != 3:
            print(f"Error: Invalid lipid spec '{spec}'. Use format: NAME:TOP:BOTTOM")
            return 1
        name = parts[0].upper()
        try:
            count_top = int(parts[1])
            count_bottom = int(parts[2])
        except ValueError:
            print(f"Error: Invalid counts in '{spec}'")
            return 1
        lipids[name] = (count_top, count_bottom)

    # Build
    print("Building membrane with composition:")
    for name, (top, bot) in lipids.items():
        print(f"  {name}: {top} top, {bot} bottom")

    print(f"Box: {args.box[0]} x {args.box[1]} x {args.box[2]} Å")

    membrane = MembraneBuilder.quick_build(
        lipids=lipids,
        box_size=(args.box[0], args.box[1]),
        box_height=args.box[2],
        bending_modulus=args.bending_modulus,
        seed=args.seed,
    )

    # Output
    output_dir = args.output or "output"
    os.makedirs(output_dir, exist_ok=True)

    pdb_path = os.path.join(output_dir, "membrane.pdb")
    membrane.write_pdb(pdb_path)
    print(f"\nWrote: {pdb_path}")

    gro_path = os.path.join(output_dir, "membrane.gro")
    membrane.write_gro(gro_path)
    print(f"Wrote: {gro_path}")

    if membrane.properties:
        print("\n" + membrane.properties.summary())

    return 0


def cmd_list_lipids(args):
    """List all available lipids."""
    library = LipidLibrary()

    if args.category:
        try:
            cat = LipidCategory[args.category.upper()]
            lipids = library.get_by_category(cat)
            print(f"\n{cat.name} lipids:\n")
        except KeyError:
            print(f"Error: Unknown category '{args.category}'")
            print("Available categories:", ", ".join(c.name for c in LipidCategory))
            return 1
    else:
        lipids = list(library)
        print(f"\nAll available lipids ({len(lipids)} total):\n")

    # Group by category
    categories = {}
    for lip in lipids:
        cat = lip.category.name
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(lip)

    for cat_name, cat_lipids in sorted(categories.items()):
        print(f"[{cat_name}]")
        for lip in sorted(cat_lipids, key=lambda x: x.name):
            charge_str = f" ({lip.charge:+.0f})" if lip.charge != 0 else ""
            print(f"  {lip.name:6s} - {lip.full_name}{charge_str}")
        print()

    return 0


def cmd_info(args):
    """Show detailed information about a lipid."""
    library = LipidLibrary()
    lipid = library.get(args.name)

    if lipid is None:
        print(f"Error: Unknown lipid '{args.name}'")
        print("\nDid you mean one of these?")
        matches = library.search(args.name)
        for lip in matches[:5]:
            print(f"  {lip.name}")
        return 1

    print()
    print("=" * 60)
    print(f"LIPID: {lipid.name}")
    print("=" * 60)
    print()
    print(f"Full name:     {lipid.full_name}")
    print(f"Category:      {lipid.category.name}")
    print()
    print("STRUCTURAL PROPERTIES:")
    print(f"  Molecular weight:    {lipid.molecular_weight:.1f} Da")
    print(f"  Charge:              {lipid.charge:+.1f} e")
    print(f"  Tail carbons:        {lipid.tail_carbons[0]}:{lipid.tail_unsaturations[0]}, "
          f"{lipid.tail_carbons[1]}:{lipid.tail_unsaturations[1]}")
    print(f"  Headgroup atoms:     {lipid.headgroup_atoms}")
    print(f"  Anchor atom:         {lipid.anchor_atom}")
    print()
    print("PHYSICAL PROPERTIES:")
    print(f"  Area per lipid:      {lipid.area_per_lipid:.1f} Å²")
    print(f"  Thickness contrib:   {lipid.thickness_contribution:.1f} Å")
    print(f"  Bending modulus:     {lipid.bending_modulus:.1f} kT")
    print()
    print("DESCRIPTION:")
    print(f"  {lipid.description}")
    print()
    print("=" * 60)

    return 0


def cmd_calculate(args):
    """Calculate membrane physical properties."""
    # Parse composition
    composition = {}
    for spec in args.lipids:
        parts = spec.split(":")
        if len(parts) != 2:
            print(f"Error: Invalid spec '{spec}'. Use format: NAME:COUNT")
            return 1
        name = parts[0].upper()
        try:
            count = int(parts[1])
        except ValueError:
            print(f"Error: Invalid count in '{spec}'")
            return 1
        composition[name] = count

    # Calculate
    physics = MembranePhysics(temperature=args.temperature)

    try:
        props = physics.calculate(
            composition=composition,
            box_area=args.box_area,
            bending_modulus_override=args.bending_modulus,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(props.summary())

    return 0


def cmd_generate_config(args):
    """Generate a sample configuration file."""
    config = MembraneConfig.create_simple(
        lipids={
            "POPC": (64, 64),
            "POPE": (16, 16),
            "CHOL": (32, 32),
        },
        box_size=(80.0, 80.0, 120.0),
        project_name="sample_membrane",
        output_dir="output/sample",
    )

    output_path = args.output or "membrane_config.yaml"
    config.to_yaml(output_path)
    print(f"Generated sample configuration: {output_path}")
    print("\nEdit this file to customize your membrane, then run:")
    print(f"  membrane-builder build --config {output_path}")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="membrane-builder",
        description="Build lipid bilayer membranes for molecular simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  membrane-builder quick --lipids POPC:64:64 CHOL:32:32 --box 80 80 120
  membrane-builder build --config my_membrane.yaml
  membrane-builder list-lipids
  membrane-builder info POPC
  membrane-builder calculate --lipids POPC:64 POPE:32 --box-area 6400
  membrane-builder generate-config --output my_config.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build membrane from config file")
    build_parser.add_argument("--config", "-c", required=True, help="Configuration YAML file")
    build_parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    build_parser.add_argument("--use-templates", action="store_true",
                              help="Use template conformer files")
    build_parser.add_argument("--templates-dir", help="Directory containing lipid templates")
    build_parser.set_defaults(func=cmd_build)

    # Quick build command
    quick_parser = subparsers.add_parser("quick", help="Quick build from command line")
    quick_parser.add_argument("--lipids", "-l", nargs="+", required=True,
                              help="Lipid specs: NAME:TOP:BOTTOM (e.g., POPC:64:64)")
    quick_parser.add_argument("--box", "-b", type=float, nargs=3, default=[80, 80, 120],
                              metavar=("LX", "LY", "LZ"), help="Box dimensions in Å")
    quick_parser.add_argument("--bending-modulus", "-k", type=float,
                              help="Override bending modulus (kT)")
    quick_parser.add_argument("--output", "-o", help="Output directory")
    quick_parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    quick_parser.set_defaults(func=cmd_quick)

    # List lipids command
    list_parser = subparsers.add_parser("list-lipids", help="List available lipids")
    list_parser.add_argument("--category", "-c", help="Filter by category")
    list_parser.set_defaults(func=cmd_list_lipids)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show lipid information")
    info_parser.add_argument("name", help="Lipid name (e.g., POPC)")
    info_parser.set_defaults(func=cmd_info)

    # Calculate command
    calc_parser = subparsers.add_parser("calculate", help="Calculate membrane properties")
    calc_parser.add_argument("--lipids", "-l", nargs="+", required=True,
                             help="Lipid specs: NAME:COUNT (e.g., POPC:128)")
    calc_parser.add_argument("--box-area", type=float, help="Box area in Å²")
    calc_parser.add_argument("--bending-modulus", "-k", type=float,
                             help="Override bending modulus (kT)")
    calc_parser.add_argument("--temperature", "-T", type=float, default=310.15,
                             help="Temperature in Kelvin")
    calc_parser.set_defaults(func=cmd_calculate)

    # Generate config command
    gen_parser = subparsers.add_parser("generate-config", help="Generate sample config file")
    gen_parser.add_argument("--output", "-o", help="Output file path")
    gen_parser.set_defaults(func=cmd_generate_config)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
