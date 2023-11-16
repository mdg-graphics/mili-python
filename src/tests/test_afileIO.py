#!/usr/bin/env python3

"""
SPDX-License-Identifier: (MIT)
"""

import logging
import os
import unittest

from mili import afileIO, datatypes

dir_path = os.path.dirname(os.path.realpath(__file__))

''' The AFileReader tests check that we can traverse the files without causing any exceptions. '''
class AFileReaderV3(unittest.TestCase):
  def test_serial_single_statefile( self ):
    data_dir = os.path.join( dir_path,'data','v3','serial_t' )
    base_name = 'd3samp6.plt'
    afile_name = os.path.join( data_dir, afileIO.afiles_by_base( data_dir, base_name )[0] )
    tfile_name = os.path.join( data_dir, base_name + 'T' )
    with open(afile_name,'rb') as af:
      with open(tfile_name,'rb') as tf:
        reader = afileIO.AFileReader()
        reader.read( af, tf )
        for dir_type in datatypes.DirectoryDecl.Type:
          reader.read_dirs( af, dir_type )

class AFileReaderV2(unittest.TestCase):
  def test_serial_single_statefile( self ):
    data_dir = os.path.join( dir_path,'data','serial','sstate' )
    base_name = 'd3samp6.plt'
    afile_name = os.path.join( data_dir, afileIO.afiles_by_base( data_dir, base_name )[0] )
    with open(afile_name,'rb') as af:
      reader = afileIO.AFileReader()
      reader.read( af, None )
      for dir_type in datatypes.DirectoryDecl.Type:
        reader.read_dirs( af, dir_type )

  def test_serial_multi_statefile( self ):
    data_dir = os.path.join( dir_path,'data','serial','mstate' )
    base_name = 'd3samp6.plt_c'
    afile_name = os.path.join( data_dir, afileIO.afiles_by_base( data_dir, base_name )[0] )
    with open(afile_name,'rb') as af:
        reader = afileIO.AFileReader()
        reader.read( af, None )
        for dir_type in datatypes.DirectoryDecl.Type:
          reader.read_dirs( af, dir_type )

class ParseDatabase(unittest.TestCase):
  
  def test_v3_serial_single_statefile_parse( self ):
    afiles, rvals = afileIO.parse_database( os.path.join( dir_path,'data','v3','serial_t','d3samp6.plt' ) )
    afile_v3 = afiles[0]
    self.assertTrue( rvals[0] )
    self.assertEqual( afile_v3.file_version, 3 )

  def test_v3_serial_state_maps( self ):
    afiles, rvals = afileIO.parse_database( os.path.join( dir_path,'data','v3','serial_t','d3samp6.plt' ) )
    afile_v3 = afiles[0]
    self.assertTrue( rvals[0] )
    desireds = [( 0, 0.0), ( 17460, 9.999999747378752e-06), ( 34920, 1.9999999494757503e-05), ( 52380, 2.9999999242136255e-05), ( 69840, 3.9999998989515007e-05), ( 87300, 4.999999873689376e-05), ( 104760, 5.999999848427251e-05), ( 122220, 7.000000186963007e-05), ( 139680, 7.999999797903001e-05), ( 157140, 9.000000136438757e-05), ( 174600, 9.999999747378752e-05), ( 192060, 0.00011000000085914508), ( 209520, 0.00011999999696854502), ( 226980, 0.00013000000035390258), ( 244440, 0.00014000000373926014), ( 261900, 0.0001500000071246177), ( 279360, 0.00015999999595806003), ( 296820, 0.00016999999934341758), ( 314280, 0.00018000000272877514), ( 331740, 0.0001900000061141327), ( 349200, 0.00019999999494757503), ( 366660, 0.0002099999983329326), ( 384120, 0.00022000000171829015), ( 401580, 0.0002300000051036477), ( 419040, 0.00023999999393709004), ( 436500, 0.0002500000118743628), ( 453960, 0.00026000000070780516), ( 471420, 0.0002699999895412475), ( 488880, 0.0002800000074785203), ( 506340, 0.0002899999963119626), ( 523800, 0.0003000000142492354), ( 541260, 0.0003100000030826777), ( 558720, 0.00031999999191612005), ( 576180, 0.00033000000985339284), ( 593640, 0.00033999999868683517), ( 611100, 0.0003499999875202775), ( 628560, 0.0003600000054575503), ( 646020, 0.0003699999942909926), ( 663480, 0.0003800000122282654), ( 680940, 0.00039000000106170774), ( 698400, 0.00039999998989515007), ( 715860, 0.00041000000783242285), ( 733320, 0.0004199999966658652), ( 750780, 0.0004299999854993075), ( 768240, 0.0004400000034365803), ( 785700, 0.00044999999227002263), ( 803160, 0.0004600000102072954), ( 820620, 0.00046999999904073775), ( 838080, 0.0004799999878741801), ( 855540, 0.0004900000058114529), ( 873000, 0.0005000000237487257), ( 890460, 0.0005099999834783375), ( 907920, 0.0005200000014156103), ( 925380, 0.0005300000193528831), ( 942840, 0.000539999979082495), ( 960300, 0.0005499999970197678), ( 977760, 0.0005600000149570405), ( 995220, 0.0005699999746866524), ( 1012680, 0.0005799999926239252), ( 1030140, 0.000590000010561198), ( 1047600, 0.0006000000284984708), ( 1065060, 0.0006099999882280827), ( 1082520, 0.0006200000061653554), ( 1099980, 0.0006300000241026282), ( 1117440, 0.0006399999838322401), ( 1134900, 0.0006500000017695129), ( 1152360, 0.0006600000197067857), ( 1169820, 0.0006699999794363976), ( 1187280, 0.0006799999973736703), ( 1204740, 0.0006900000153109431), ( 1222200, 0.000699999975040555), ( 1239660, 0.0007099999929778278), ( 1257120, 0.0007200000109151006), ( 1274580, 0.0007300000288523734), ( 1292040, 0.0007399999885819852), ( 1309500, 0.000750000006519258), ( 1326960, 0.0007600000244565308), ( 1344420, 0.0007699999841861427), ( 1361880, 0.0007800000021234155), ( 1379340, 0.0007900000200606883), ( 1396800, 0.0007999999797903001), ( 1414260, 0.0008099999977275729), ( 1431720, 0.0008200000156648457), ( 1449180, 0.0008299999753944576), ( 1466640, 0.0008399999933317304), ( 1484100, 0.0008500000112690032), ( 1501560, 0.000859999970998615), ( 1519020, 0.0008699999889358878), ( 1536480, 0.0008800000068731606), ( 1553940, 0.0008900000248104334), ( 1571400, 0.0008999999845400453), ( 1588860, 0.000910000002477318), ( 1606320, 0.0009200000204145908), ( 1623780, 0.0009299999801442027), ( 1641240, 0.0009399999980814755), ( 1658700, 0.0009500000160187483), ( 1676160, 0.0009599999757483602), ( 1693620, 0.0009699999936856329), ( 1711080, 0.0009800000116229057), ( 1728540, 0.0009899999713525176), ( 1746000, 0.0010000000474974513) ]
    for result, desired in zip(afile_v3.smaps, desireds):
      self.assertEqual( result.file_offset, desired[0] )
      self.assertEqual( result.time, desired[1] )

  def test_v3_parallel_single_statefile_parse( self ):
    afiles, rvals = afileIO.parse_database( os.path.join( dir_path,'data','v3','parallel_t','d3samp6.plt' ) )
    for afile, rval in zip(afiles,rvals):
      self.assertTrue( rval )
      self.assertEqual( afile.file_version, 3 )

  def test_v3_no_statefile_parse( self ):
    afiles, rvals = afileIO.parse_database( os.path.join( dir_path,'data','v3','no_tfile','d3samp6.plt' ) )
    for afile, rval in zip(afiles,rvals):
      self.assertTrue( rval )
      self.assertEqual( afile.file_version, 3 )

  def test_v2_serial_single_statefile_parse( self ):
    afiles, rvals = afileIO.parse_database( os.path.join( dir_path,'data','serial','sstate','d3samp6.plt' ) )
    afile_v2 = afiles[ 0 ]
    self.assertTrue( rvals[0] )
    self.assertEqual( afile_v2.file_version, 2 )

  def test_v2_serial_multi_statefile_parse( self ):
    afiles, rvals = afileIO.parse_database( os.path.join( dir_path,'data','serial','mstate','d3samp6.plt_c' ) )
    afile_v2 = afiles[ 0 ]
    self.assertTrue( rvals[0] )
    self.assertEqual( afile_v2.file_version, 2 )

class AFileWriter(unittest.TestCase):
  data_dir = os.path.join( dir_path,'data','serial','sstate' )
  base_name = 'd3samp6.plt'
  new_base_name = 'new_d3samp6.plt'

  def setUp(self):
    # Read in A File.
    afiles, rvals = afileIO.parse_database( os.path.join( AFileWriter.data_dir, AFileWriter.base_name ) )
    afile = afiles[0]
    self.assertTrue( rvals[0] )

    # Write out same A File.
    rvals = afileIO.write_database( afile, os.path.join(AFileWriter.data_dir, AFileWriter.new_base_name) )
    self.assertEqual( rvals[0], 0 )

  def tearDown(self):
    # Delete A files
    os.remove( f"{os.path.join(AFileWriter.data_dir, AFileWriter.new_base_name)}A" )

  def test_read_afile(self):
    # Test that we can read in new A file
    afiles, rvals = afileIO.parse_database( os.path.join(AFileWriter.data_dir, AFileWriter.new_base_name) )
    afile = afiles[0]
    self.assertTrue( rvals[0] )

    # Read in original A file and compare with new one
    afiles, rvals = afileIO.parse_database( os.path.join(AFileWriter.data_dir, AFileWriter.base_name) )
    original = afiles[0]
    self.assertTrue( rvals[0] )

    self.assertEqual( afile, original )

if __name__ == "__main__":
    unittest.main()
