#SPDX-License-Identifier: (MIT)
import argparse
import logging
import mili.afileIO as afileIO

def main(args):
  parser = argparse.ArgumentParser()
  args = parser.parse_args(args)
  logging.basicConfig( filename=args.outfile, level=logging.INFO )
  logging.info(f"discovering afiles using basename '{args.basefile}' in directory '{args.directory}':")
  afiles = afileIO.afiles_by_base( args.directory, args.basefile, args.whitelist )
  logging.info("  \n".join(afiles))

  verifier = afileIO.AFileVerifier()
  for afile in afiles:
    with open(afile,'r') as fin:
      verifier.read( fin )

