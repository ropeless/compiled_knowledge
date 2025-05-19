"""
This is an example Python script for copying Ace to CK.
"""
from ck.pgm_compiler.ace import copy_ace_to_default_location


SOURCE_ACE: str = r'C:\Research\Ace\ace_v3.0_windows'


def main() -> None:
    copy_ace_to_default_location(SOURCE_ACE)


if __name__ == "__main__":
    main()
