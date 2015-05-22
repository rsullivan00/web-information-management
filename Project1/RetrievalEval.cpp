/*==========================================================================
 * Copyright (c) 2001 Carnegie Mellon University.  All Rights Reserved.
 *
 * Use of the Lemur Toolkit for Language Modeling and Information Retrieval
 * is subject to the terms of the software license set forth in the LICENSE
 * file included with this software, and also available at
 * http://www.lemurproject.org/license.html
 *
 *==========================================================================
*/

/*! \page  Retrieval Evaluation Application within light Lemur toolkit

This application runs retrieval experiments to evaluate different retrieval models 

Usage: RetrievalEval parameter_file

Please refor to the namespace LocalParameter for setting the parameters within the parameter_file

 */


#include "common_headers.hpp"
#include "IndexManager.hpp"
#include "BasicDocStream.hpp"
#include "Param.hpp"
#include "String.hpp"
#include "IndexedReal.hpp"
#include "ScoreAccumulator.hpp"
#include "ResultFile.hpp"
#include "TextQueryRep.hpp"

using namespace lemur::api;

namespace LocalParameter{
  std::string databaseIndex; // the index of the documents
  std::string queryStream;   // the file of query stream
  std::string resultFile;    // the name of the result file
  std::string weightScheme;  // the weighting scheme
  int resultCount;           // the number of top ranked documents to return for each query
  void get() {
    // the string with quotes are the actual variable names to use for specifying the parameters
    databaseIndex    = ParamGetString("index"); 
    queryStream      = ParamGetString("query");
    resultFile       = ParamGetString("result","res");
    weightScheme     = ParamGetString("weightScheme","RawTF");
    resultCount      = ParamGetInt("resultCount", 100); 
  }    
};

void GetAppParam() 
{
  LocalParameter::get();
}

// compute the weight of a matched term
double computeRawTFWeight(int docID,
		     int termID, 
		     int docTermFreq, 
		     double qryTermWeight,
		     Index *ind)
{
  //implementation of raw TF weighting scheme
  return docTermFreq*qryTermWeight;
}


// compute the weight of a matched term
double computeRawTFIDFWeight(int docID,
		     int termID, 
		     int docTermFreq, 
		     double qryTermWeight,
		     Index *ind)
{
  return docTermFreq*log2(ind->docCount()/ind->docCount(termID))*qryTermWeight;
}


// compute the weight of a matched term
double computeLogTFIDFWeight(int docID,
		     int termID, 
		     int docTermFreq, 
		     double qryTermWeight,
		     Index *ind)
{
  return (log2(docTermFreq) + 1)*log2(ind->docCount()/ind->docCount(termID))*qryTermWeight;
}

// compute the weight of a matched term
double computeOkapiWeight(int docID,
		     int termID, 
		     int docTermFreq, 
		     double qryTermWeight,
		     Index *ind)
{
  /* Ugly, but removes necessity of short-lived variables */
  return (docTermFreq/(docTermFreq + 0.5 + 1.5*(ind->docLength(docID)/ind->docLengthAvg())))*log2((ind->docCount()-ind->docCount(termID) + 0.5)/(ind->docCount(termID) + 0.5))*((8 + qryTermWeight)/(7 + qryTermWeight));
}


double computeCustomWeight(int docID,
		     int termID, 
		     int docTermFreq, 
		     double qryTermWeight,
		     Index *ind)
{
  return pow(qryTermWeight,2)*docTermFreq/ind->docCount(termID);
}



// compute the adjusted score
double computeAdjustedScore(double origScore, // the score from the accumulator
			    int docID, // doc ID
			    Index *ind) // index
{
  //Do nothing now
  return origScore;
}


// compute the adjusted score
double computeCustomAdjustedScore(double origScore, // the score from the accumulator
			    int docID, // doc ID
			    Index *ind) // index
{
  return origScore/ind->docLength(docID);
}


void ComputeQryArr(Document *qryDoc, double *qryArr, Index *ind){
  //compute the array representation of query; it is the frequency representation for the original query 
  for (int t=1; t<=ind->termCountUnique(); t++) {
    qryArr[t]=0;
  }
  
  qryDoc->startTermIteration();
  while (qryDoc->hasMore()) {
    const Term *qryTerm = qryDoc->nextTerm();
    int qryTermID = ind->term(qryTerm->spelling());
    qryArr[qryTermID] ++;
  }
}



void Retrieval(double *qryArr, IndexedRealVector &results, Index *ind){
  //retrieve documents with respect to the array representation of the query

  lemur::retrieval::ArrayAccumulator scoreAccumulator(ind->docCount());

  scoreAccumulator.reset();
  for (int t=1; t<=ind->termCountUnique();t++) {
    if (qryArr[t]>0) {      
      // fetch inverted entries for a specific query term
      DocInfoList *docList = ind->docInfoList(t);

      // iterate over all individual documents 
      docList->startIteration();
      while (docList->hasMore()) {
	DocInfo *matchInfo = docList->nextEntry();
	// for each matched term, calculated the evidence

	double wt;

	if (strcmp(LocalParameter::weightScheme.c_str(),"RawTF")==0) {
	  wt = computeRawTFWeight(matchInfo->docID(),  // doc ID
				  t, // term ID
				  matchInfo->termCount(), // freq of term t in this doc
				  qryArr[t], // freq of term t in the query
				  ind);	  
	}else if (strcmp(LocalParameter::weightScheme.c_str(),"RawTFIDF")==0) {
	  wt = computeRawTFIDFWeight(matchInfo->docID(),  // doc ID
				  t, // term ID
				  matchInfo->termCount(), // freq of term t in this doc
				  qryArr[t], // freq of term t in the query
				  ind);	  
	}else if (strcmp(LocalParameter::weightScheme.c_str(),"LogTFIDF")==0) {
	  wt = computeLogTFIDFWeight(matchInfo->docID(),  // doc ID
				  t, // term ID
				  matchInfo->termCount(), // freq of term t in this doc
				  qryArr[t], // freq of term t in the query
				  ind);	  
	}else if (strcmp(LocalParameter::weightScheme.c_str(),"Okapi")==0) {
	  wt = computeOkapiWeight(matchInfo->docID(),  // doc ID
				  t, // term ID
				  matchInfo->termCount(), // freq of term t in this doc
				  qryArr[t], // freq of term t in the query
				  ind);	  
	}else if (strcmp(LocalParameter::weightScheme.c_str(),"Custom")==0){
	  wt = computeCustomWeight(matchInfo->docID(),  // doc ID
				  t, // term ID
				  matchInfo->termCount(), // freq of term t in this doc
				  qryArr[t], // freq of term t in the query
				  ind);	  
	}else{
	  cerr<<"The weighting scheme of "<<LocalParameter::weightScheme.c_str()<<" is not supported"<<endl;
          exit(1);
	}
	scoreAccumulator.incScore(matchInfo->docID(),wt);  
      }
      delete docList;
    }
  }

  // Adjust the scores for the documents when it is necessary
  double s;
  int d;
  for (d=1; d<=ind->docCount(); d++) {
    if (scoreAccumulator.findScore(d,s)) {
    } else {
      s=0;
    }

    if (strcmp(LocalParameter::weightScheme.c_str(),"RawTF")==0) {
      results.PushValue(d, computeAdjustedScore(s, // the score from the accumulator
						d, // doc ID
						ind)); // index
    }else if (strcmp(LocalParameter::weightScheme.c_str(),"RawTFIDF")==0) {
      results.PushValue(d, computeAdjustedScore(s, // the score from the accumulator
						d, // doc ID
						ind)); // index
    }else if (strcmp(LocalParameter::weightScheme.c_str(),"LogTFIDF")==0) {
      results.PushValue(d, computeAdjustedScore(s, // the score from the accumulator
						d, // doc ID
						ind)); // index
    }else if (strcmp(LocalParameter::weightScheme.c_str(),"Okapi")==0) {
      results.PushValue(d, computeAdjustedScore(s, // the score from the accumulator
						d, // doc ID
						ind)); // index
    }else if (strcmp(LocalParameter::weightScheme.c_str(),"Custom")==0){
      results.PushValue(d, computeCustomAdjustedScore(s, // the score from the accumulator
						      d, // doc ID
					      ind)); // index     
    }else{
      cerr<<"The weighting scheme of "<<LocalParameter::weightScheme.c_str()<<" is not supported"<<endl;
      exit(1);
    }
  }
}


/// A retrieval evaluation program
int AppMain(int argc, char *argv[]) {
  

  //Step 1: Open the index file
  Index  *ind;

  try {
    ind  = IndexManager::openIndex(LocalParameter::databaseIndex);
  } 
  catch (Exception &ex) {
    ex.writeMessage();
    throw Exception("RelEval", "Can't open index, check parameter index");
  }

  //Step 2: Open the query file
  DocStream *qryStream;
  try {
    qryStream = new lemur::parse::BasicDocStream(LocalParameter::queryStream);
  } 
  catch (Exception &ex) {
    ex.writeMessage(cerr);
    throw Exception("RetEval", 
                    "Can't open query file, check parameter textQuery");
  }

  //Step 3: Create the result file
  ofstream result(LocalParameter::resultFile.c_str());
  ResultFile resultFile(1);
  resultFile.openForWrite(result, *ind);


  // go through each query
  
  qryStream->startDocIteration();
  while (qryStream->hasMore()) {
    Document *qryDoc = qryStream->nextDoc();
    const char *queryID = qryDoc->getID();
    cout << "query: "<< queryID <<endl;

    double *queryArr = new double[ind->termCountUnique()+1];  //the array that contains the weights of query terms; for orignial query 
    ComputeQryArr(qryDoc,queryArr, ind); 

    IndexedRealVector results(ind->docCount());
    results.clear();

    Retrieval(queryArr,results,ind);

    results.Sort();
    resultFile.writeResults(queryID, &results, LocalParameter::resultCount);

    delete queryArr;
  }



  result.close();
  delete qryStream;
  delete ind;
  return 0;
}

