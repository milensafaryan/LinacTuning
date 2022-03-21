#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Tools.h"
#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/TMVARegGui.h"


using namespace TMVA;

void regression( TString myMethodList = "", TString outfileName="tmvareg_testout.root", TString outdir="dataset" )
{
   //     mylinux~> root -l TMVARegression.C\(\"myMethod1,myMethod2,myMethod3\"\)
   //
   TMVA::Tools::Instance();

   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );
   TMVA::Factory *factory = new TMVA::Factory( "TMVARegression", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:AnalysisType=Regression" );


   TMVA::DataLoader *dataloader=new TMVA::DataLoader(outdir);
   dataloader->AddTarget( "RFQPAH_R", "RFQ phase", "deg", 'D' );
   dataloader->AddTarget( "RFBPAH_R", "buncher phase", "deg", 'D' );
   dataloader->AddTarget( "V5QSET_R", "Tank 5 phase", "deg", 'D' );

   dataloader->AddSpectator( "RFQPAH_S", "RFQ phase set point", "deg", 'D' );
   dataloader->AddSpectator( "RFBPAH_S", "buncher phase set point", "deg", 'D' );
   dataloader->AddSpectator( "V5QSET_S", "Tank 5 phase set point", "deg", 'D' );

   dataloader->AddVariable( "LMSM",     "LMSM", "cnt", 'D');
   dataloader->AddVariable( "D7TOR_R",  "D7TOR", "mA", 'D' );
   dataloader->AddVariable( "TO1IN_R",  "TO1IN", "mA", 'D' );
   dataloader->AddVariable( "TO3IN_R",  "TO3IN", "mA", 'D' );
   dataloader->AddVariable( "TO5OUT_R", "TO5OUT", "mA", 'D' );
   dataloader->AddVariable( "D00LM_R",  "D00LM", "cnt", 'D');
   
   for(int i=0; i<7; i++){
     for(int j=0; j<4; j++){
       dataloader->AddVariable(Form("D%i%iLM_R",i+1,j+1), Form("D%i%iLM",i+1,j+1), "cnt", 'D');
     }
   }
   

   TFile *input(0);
   TString fname = "../06212021/3phase.root";
   if (!gSystem->AccessPathName( fname )) {
      input = TFile::Open( fname ); // check if file in local directory exists
   }
   else {
     std::cout << "ERROR: could not find data file" << std::endl;
     exit(1);
   }
   std::cout << "--- TMVARegression           : Using input file: " << input->GetName() << std::endl;

   // Register the regression tree
   TTree *regTree = (TTree*)input->Get("paramT");

   // global event weights per tree (see below for setting event-wise weights)
   Double_t regWeight  = 1.0;

   // You can add an arbitrary number of regression trees
   dataloader->AddRegressionTree( regTree, regWeight );
   //dataloader->SetWeightExpression( "var1", "Regression" );

   // Apply additional cuts on the signal and background samples (can be different)
   TCut mycut = "LMSM>0 && LMSM<40. && D7TOR_R>0. && D7TOR_R<30 && D00LM_R>0."; // for example: TCut mycut = "abs(var1)<0.5 && abs(var2-0.5)<1";

   // tell the DataLoader to use all remaining events in the trees after training for testing:
   dataloader->PrepareTrainingAndTestTree( mycut,
                                         "nTrain_Regression=5000:nTest_Regression=0:SplitMode=Random:NormMode=NumEvents:!V" );
   // Neural network (MLP)
   //factory->BookMethod( dataloader,  TMVA::Types::kMLP, "MLP", "!H:!V:VarTransform=Norm:NeuronType=tanh:NCycles=15000:HiddenLayers=N+20:TestRate=6:TrainingMethod=BFGS:Sampling=0.3:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=15:!UseRegulator");

   //DNN
   TString layoutString("Layout=RELU|50,RELU|50,RELU|50,LINEAR");   
   TString trainingStrategyString("TrainingStrategy=");   
   trainingStrategyString +="LearningRate=1e-3,Momentum=0.3,ConvergenceSteps=20,BatchSize=30,TestRepetitions=1,WeightDecay=0.0,Regularization=None,Optimizer=Adam";   
   TString nnOptions("!H:V:ErrorStrategy=SUMOFSQUARES:VarTransform=N:WeightInitialization=XAVIERUNIFORM:Architecture=CPU");
   nnOptions.Append(":");
   nnOptions.Append(layoutString);
   nnOptions.Append(":");
   nnOptions.Append(trainingStrategyString);
   
   factory->BookMethod(dataloader, TMVA::Types::kDL, "DNN_CPU", nnOptions); // NN

   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   // Evaluate all MVAs using the set of test events
   factory->TestAllMethods();

   // Evaluate and compare performance of all configured MVAs
   factory->EvaluateAllMethods();

   // --------------------------------------------------------------

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVARegression is done!" << std::endl;

   delete factory;
   delete dataloader;

   // Launch the GUI for the root macros
   if (!gROOT->IsBatch()) TMVA::TMVARegGui( outfileName );
}

int main( int argc, char** argv )
{
   regression();
   return 0;
}
