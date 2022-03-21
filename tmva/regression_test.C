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

void regression_test( TString myMethodList = "", TString outfileName="tmvareg_testout.root" )
{
   //     mylinux~> root -l TMVARegression.C\(\"myMethod1,myMethod2,myMethod3\"\)
   //
   TMVA::Tools::Instance();
   std::map<std::string,int> Use;

   // Neural Network
   Use["MLP"]             = 0;
#ifdef R__HAS_TMVACPU
   Use["DNN_CPU"] = 1;
#else
   Use["DNN_CPU"] = 0;
#endif

   std::cout << std::endl;
   std::cout << "==> Start TMVARegression" << std::endl;

   if (myMethodList != "") {
      for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;
      std::vector<TString> mlist = gTools().SplitString( myMethodList, ',' );
      for (UInt_t i=0; i<mlist.size(); i++) {
         std::string regMethod(mlist[i].Data());
         if (Use.find(regMethod) == Use.end()) {
            std::cout << "Method \"" << regMethod << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
            for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) std::cout << it->first << " ";
            std::cout << std::endl;
            return;
         }
         Use[regMethod] = 1;
      }
   }

   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );
   TMVA::Factory *factory = new TMVA::Factory( "TMVARegression", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:AnalysisType=Regression" );


   TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset");
   dataloader->AddVariable( "RFQPAH_R", "RFQ phase", "units", 'D' );
   dataloader->AddVariable( "RFBPAH_R", "buncher phase", "units", 'D' );
   dataloader->AddVariable( "V5QSET_R", "tank5 phase", "units", 'D' );

   dataloader->AddSpectator( "RFQPAH_S", "RFQ phase Set", "degrees", 'D' );
   dataloader->AddSpectator( "RFBPAH_S", "buncher phase Set", "degrees", 'D' );
   dataloader->AddSpectator( "V5QSET_S", "Tank 5 phase", "degrees", 'D' );

   dataloader->AddSpectator( "TO1IN_R",  "to1in", "units", 'D' );
   dataloader->AddSpectator( "TO5OUT_R",  "to5out", "units", 'D' );
   dataloader->AddSpectator( "TO3IN_R",  "to3in", "units", 'D' );

   // Add the variable carrying the regression target
   dataloader->AddTarget( "LMSM" );
   dataloader->AddTarget( "D7TOR_R" );
   dataloader->AddTarget( "TO1IN_R",  "to1in", "mA", 'D' );
   dataloader->AddTarget( "TO3IN_R",  "to3in", "mA", 'D' );
   dataloader->AddTarget( "TO5OUT_R", "to5out", "mA", 'D' );
   dataloader->AddTarget( "D00LM_R",  "d00lm", "", 'D');
   
   for(int i=0; i<7; i++){
     for(int j=0; j<4; j++){
       dataloader->AddTarget(Form("D%i%iLM_R",i+1,j+1), Form("d%i%ilm",i+1,j+1), "", 'D');
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
   if (Use["MLP"])
      factory->BookMethod( dataloader,  TMVA::Types::kMLP, "MLP", "!H:!V:VarTransform=Norm:NeuronType=tanh:NCycles=15000:HiddenLayers=N+20:TestRate=6:TrainingMethod=BFGS:Sampling=0.3:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=15:!UseRegulator" );

   if (Use["DNN_CPU"]) {

      TString layoutString("Layout=RELU|50,RELU|50,RELU|50,LINEAR");

      TString trainingStrategyString("TrainingStrategy=");

      trainingStrategyString +="LearningRate=1e-3,Momentum=0.3,ConvergenceSteps=20,BatchSize=30,TestRepetitions=1,WeightDecay=0.0,Regularization=None,Optimizer=Adam";

      TString nnOptions("!H:V:ErrorStrategy=SUMOFSQUARES:VarTransform=N:WeightInitialization=XAVIERUNIFORM:Architecture=CPU");
      nnOptions.Append(":");
      nnOptions.Append(layoutString);
      nnOptions.Append(":");
      nnOptions.Append(trainingStrategyString);

      factory->BookMethod(dataloader, TMVA::Types::kDL, "DNN_CPU", nnOptions); // NN
   }

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
   // Select methods (don't look at this code - not of interest)
   TString methodList;
   for (int i=1; i<argc; i++) {
      TString regMethod(argv[i]);
      if(regMethod=="-b" || regMethod=="--batch") continue;
      if (!methodList.IsNull()) methodList += TString(",");
      methodList += regMethod;
   }
   regression_test(methodList);
   return 0;
}
