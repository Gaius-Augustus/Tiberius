#!/usr/bin/env python3
import argparse
from tiberius import parseCmd, run_tiberius

def main():    
    args = parseCmd()
    run_tiberius(args)

if __name__ == '__main__':
    main()
