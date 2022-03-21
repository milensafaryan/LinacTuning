/// The methods to be used can be switched on and off by means of booleans, or
/// via the prompt command, for example:
///
///     root -l TMVARegression.C\(\"LD,MLP\"\)
///
/// (note that the backslashes are mandatory)

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

void regression_all_losses_tors( TString myMethodList = "", TString outfileName = "tmvareg_3ph.root" )
{
   // The explicit loading of the shared libTMVA is done in TMVAlogon.C, defined in .rootrc
   // if you use your private .rootrc, or run from a different directory, please copy the
   // corresponding lines from .rootrc

   // methods to be processed can be given as an argument; use format:
   //
   //  root -l TMVARegression.C\(\"myMethod1,myMethod2,myMethod3\"\)
   //

   //---------------------------------------------------------------
   // This loads the library
   TMVA::Tools::Instance();

   // Default MVA methods to be trained + tested
   std::map<std::string,int> Use;

   // Neural Network
   Use["MLP"]             = 1;
#ifdef R__HAS_TMVACPU
   Use["DNN_CPU"] = 1;
#else
   Use["DNN_CPU"] = 0;
#endif
   //
   // ---------------------------------------------------------------

   std::cout << std::endl;
   std::cout << "==> Start TMVARegression" << std::endl;

   // Select methods (don't look at this code - not of interest)
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

   // --------------------------------------------------------------------------------------------------

   // Here the preparation phase begins

   // Create a new root output file
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

   TMVA::Factory *factory = new TMVA::Factory( "TMVARegression", outputFile,
                                               "!V:!Silent:Color:DrawProgressBar:AnalysisType=Regression" );
   TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset_3ph");
   // If you wish to modify default settings
   // (please check "src/Config.h" to see all available global options)
   //
   //     (TMVA::gConfig().GetVariablePlotting()).fTimesRMS = 8.0;
   //     (TMVA::gConfig().GetIONames()).fWeightFileDir = "myWeightDirectory";

   dataloader->AddVariable( "RFQPAH_R", "RFQ phase", "degrees", 'D'  );
   dataloader->AddVariable( "RFBPAH_R", "buncher phase", "degrees", 'D' );
   dataloader->AddVariable( "V5QSET_R", "Tank 5 phase", "degrees", 'D' );
   
   dataloader->AddSpectator( "RFQPAH_S", "RFQ phase Set", "degrees", 'D' );
   dataloader->AddSpectator( "RFBPAH_S", "buncher phase Set", "degrees", 'D' );
   dataloader->AddSpectator( "V5QSET_S", "Tank 5 phase", "degrees", 'D' );

   dataloader->AddTarget( "TO1IN_R",  "to1in", "mA", 'D' );
   dataloader->AddTarget( "TO3IN_R",  "to3in", "mA", 'D' );
   dataloader->AddTarget( "TO5OUT_R", "to5out", "mA", 'D' );
   dataloader->AddTarget( "D7TOR_R",  "d7tor", "mA", 'D' );
   dataloader->AddTarget( "LMSM",  "dlmsm", "", 'D');
   dataloader->AddTarget( "D00LM_R",  "d00lm", "", 'D');

   /*
   for(int i=0; i<7; i++){
     for(int j=0; j<4; j++){
       dataloader->AddTarget(Form("D%i%iLM_R",i+1,j+1), Form("d%i%ilm",i+1,j+1), "", 'D');
     }
   }
   */
   
   // Read training and test data (see TMVAClassification for reading ASCII files)
   // load the signal and background event samples from ROOT trees
   TFile *input(0);
   TString fname = "../06212021/3phase.root";
   if (!gSystem->AccessPathName( fname )) {
      input = TFile::Open( fname ); // check if file in local directory exists
   }
   if (!input) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }
   std::cout << "--- TMVARegression           : Using input file: " << input->GetName() << std::endl;

   // Register the regression tree

   TTree *regTree = (TTree*)input->Get("paramT");

   // global event weights per tree (see below for setting event-wise weights)
   Double_t regWeight  = 1.0;

   // You can add an arbitrary number of regression trees
   dataloader->AddRegressionTree( regTree, regWeight );

   // Apply additional cuts on the signal and background samples (can be different)
   TCut mycut = "LMSM>0 && LMSM<40. && D00LM_R>0 && D7TOR_R>0. && D7TOR_R<30"; // for example: TCut mycut = "abs(var1)<0.5 && abs(var2-0.5)<1";
   //TCut mycut = "";

   // tell the DataLoader to use all remaining events in the trees after training for testing:
   dataloader->PrepareTrainingAndTestTree( mycut,
                                         "nTrain_Regression=8000:nTest_Regression=0:SplitMode=Random:NormMode=NumEvents:!V" );

   // Neural network (MLP)
   if (Use["MLP"])
      factory->BookMethod( dataloader,  TMVA::Types::kMLP, "MLP", "!H:!V:VarTransform=Norm:NeuronType=tanh:NCycles=2000:HiddenLayers=N+10,10,10:TestRate=6:TrainingMethod=BFGS:Sampling=0.3:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=15:!UseRegulator" );

   if (Use["DNN_CPU"]) {

      TString layoutString("Layout=TANH|50,TANH|50,TANH|50,LINEAR");


      TString trainingStrategyString("TrainingStrategy=");

      trainingStrategyString +="LearningRate=1e-3,Momentum=0.3,ConvergenceSteps=20,BatchSize=10,TestRepetitions=1,WeightDecay=0.0,Regularization=None,Optimizer=Adam";

      TString nnOptions("!H:V:ErrorStrategy=SUMOFSQUARES:VarTransform=G:WeightInitialization=XAVIERUNIFORM:Architecture=CPU");
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
   regression_all_losses_tors(methodList);
   return 0;
}
