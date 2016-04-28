from __future__ import division
import unittest
import hwNew
import nltk

class TestHwMethods(unittest.TestCase):

  def test_jaccard(self):
      self.assertEqual(hwNew.jaccard([1,2,0,1],[0,0,1,1]), 0.25)
  def test_idf(self):
      self.assertEqual(hwNew.BuildIdf([[0.5,0,0,0.2],[0.2,0.01,0.5,1],[0.2,0.4,0,1]]),[0.0, 0.17609125905568124, 0.47712125471966244, 0.0])


  def test_idfDummy(self):
      self.assertEqual(hwNew.BuildIdfDummy([[0,0,0,1],[0,0,1,1],[1,1,0,1]]),[1,1,1,1])
  def test_tf(self):
      freq={"this":3,"that":5}
      key ={"this":0,"that":1,"test":2}
      self.assertEqual(hwNew.tf(key,freq),[3/8,5/8,0])

  def test_feedback(self):
      nltk.data.path.append("/media/sf_shared/nltk")
      docVectors={"test1":[10]*25}
      keyPosition ={'2for': 0, 'ar': 1, 'motor': 2, 'carbon': 3, 'sharp': 4, '#50card': 5, 'econom': 6, 'plastic': 7, 'ideal': 8, 'wood': 9, 'price': 13, 'drill': 21, 'thei': 12, 'tool': 10, 'hand': 14, 'dremel,': 15, 'toolscarbon': 23, 'steelwir': 17, 'steel': 18, 'gyros,': 19, 'gyro': 20, 'flexible,': 11, 'gaug': 22, 'extrem': 16, 'rotari': 24}
      query=[1]*25
      result=[1.0,1.0,6.0,6.0,1.0 ,1.0,6.0,6.0,6.0,6.0 ,6.0,1.0,1.0,6.0,6.0 ,6.0,6.0,6.0,6.0,1.0,1.0,6.0,6.0,6.0, 6.0]
      #[('2for', 'CD'), ('ar', 'JJ'),
      #('motor', 'NN'), ('carbon', 'NN'), ('sharp', 'JJ'), ('#50card', 'NNP'), ('econom', 'NN')
      #, ('plastic', 'NN'), ('ideal', 'NN'), ('wood', 'NN'), ('price', 'NN')
      #, ('drill', 'NN'), ('thei', 'IN'), ('tool', 'JJ'), ('hand', 'NN'), ('dremel,', 'NN')
      #, ('toolscarbon', 'NN'), ('steelwir', 'NN'), ('steel', 'NN'), ('gyros,', 'NNS'), ('gyro', 'VBP')
      #, ('flexible,', 'JJ'), ('gaug', 'NN'), ('extrem', 'NN'), ('rotari', 'NN')]
      #
      realResult=hwNew.feedback(query,docVectors,keyPosition)
      self.assertEqual(realResult,result)
      self.assertEqual(realResult.count(6),17)

  def test_feedback2(self):
      nltk.data.path.append("/media/sf_shared/nltk")
      docVectors={"test1":[10]*52}
      keyPosition = {'clip': 0, '18-volt,': 1, '2for': 3, 'hook': 4, 'ar': 5, 'motor': 6, 'steelwir': 38, 'sharp': 8, 'mobil': 9, 'belt': 11, 'impact': 12, '#50card': 13, 'siw': 14, 'drill': 48, 'secur': 15, 'beltfor': 16, 'econom': 17, 'plastic': 18, 'attach': 19, 'ideal': 20, 'wood': 21, 'hookattach': 22, 'tools:': 23, 'belt,': 24, 'price': 29, 'flexible,': 26, 'thei': 27, 'jobsit': 28, 'tool': 25, 'sid': 30, 'driver': 31, 'free': 32, 'hand': 33, 'dremel,': 34, '(1)': 35, 'hilti': 39, '18-voltdur': 37, 'carbon': 7, 'materialinclud': 10, 'cordless': 40, 'sfc': 41, 'accessori': 42, 'steel': 43, 'gyros,': 44, 'gyro': 45, 'toolscarbon': 50, 'easier': 46, 'wrench': 47, 'thi': 2, 'gaug': 49, 'extrem': 36, 'rotari': 51}
      query=[1]*52
      realResult=hwNew.feedback(query,docVectors,keyPosition)
      self.assertEqual(realResult.count(6),28)

if __name__ == '__main__':
    unittest.main()
