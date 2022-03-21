using namespace std;
using namespace ROOT;

double LM[7][4]; //loss monitors
string sLM[7][4];
double TOR[5]; //toroids
string sTOR[5] = {"TO1IN","TO3IN","TO4IN","TO5OUT","D7TOR"};
double RFQp[2];
double delta = 0.1;
double mid = 163; //RFQ

TH1D* h1[7][4]; //losses vs RFQ
TH1D* h2[7]; //currents vs RFQ


//double bins[9] ={215.5,221.,226.5,231.5,237.,242.,247.5,252.5,258.};

void plot_losses_rfq(string name="./06292021/RFQ"){

  TFile* fin = new TFile(Form("%s.root",name.c_str()));
  TTree* T = (TTree*)fin->Get("paramT");
  
  /* Set branches */
  T->SetBranchAddress("RFQPAH_S",&RFQp[0]);
  T->SetBranchAddress("RFQPAH_R",&RFQp[1]);
  for(int i=0; i<7; i++){
    for(int j=0; j<4; j++){
      sLM[i][j] = Form("D%i%iLM_R",i+1,j+1);
      //cout << sLM[i][j] << endl;
      h1[i][j] = new TH1D(Form("hRFQ_D%i%i",i+1,j+1),Form(";RFQ phase [deg];D%i%iLM (avg.)",i+1,j+1),32,128.,191.);
      h1[i][j]->SetMarkerStyle(20);
      h1[i][j]->SetMarkerSize(0.7);
      h1[i][j]->SetLineWidth(2);
      T->SetBranchAddress(sLM[i][j].c_str(),&LM[i][j]);
    }
  }
  for(int i=0; i<5; i++){
    string stor =sTOR[i]+"_R";
    cout << stor << endl;
    h2[i] = new TH1D(Form("hRFQ_%s",sTOR[i].c_str()),Form(";RFQ phase [deg];%s (avg.)",sTOR[i].c_str()),32,128.,191.);
    h2[i]->SetMarkerStyle(20);
    h2[i]->SetMarkerSize(0.7);
    h2[i]->SetLineWidth(2);
    T->SetBranchAddress(stor.c_str(),&TOR[i]);
  }


  /* Loop over entries */
  for(int k=0; k<T->GetEntries(); k++){
    T->GetEntry(k);
    
    for(int i=0; i<7; i++){
      for(int j=0; j<4; j++){
	h1[i][j]->Fill(RFQp[1],LM[i][j]);
      }
    }
    for(int i=0; i<5; i++){
      h2[i]->Fill(RFQp[1],TOR[i]);
    }
    
    
  }
  std::cout << "Central bin losses " << h1[0][0]->FindBin(163.) << std::endl;
  std::cout << "Central bin tors " << h2[0]->FindBin(163.) << std::endl;
  /* Normalize histograms */
  for(int i=0; i<7; i++){
    for(int j=0; j<4; j++){
      //h1[i][j]->Scale(1./h1[i][j]->Integral());
      //h1[i][j]->Scale(1./20.);
      h1[i][j]->Scale(1./h1[i][j]->GetBinContent(18));
    }
  }
  for(int i=0; i<5; i++){
    h2[i]->Scale(1./20.);
  }
  
  //for(int b=0; b<6; b++) cout << h3[6][3]->GetBinContent(b+1) << endl;
  TCanvas* c1[7][4]; 
  for(int i=0; i<7; i++){
    for(int j=0; j<4; j++){
      c1[i][j]  = new TCanvas(Form("cLM%i%i",i+1,j+1),"",400,400);
      c1[i][j]->cd();
      h1[i][j]->Draw("phist c");
      c1[i][j]->SaveAs(Form("cRFQ_LM%i%i.png",i+1,j+1));
    }
  }
  
  TCanvas* c2[5];
  for(int i=0; i<5; i++){
    c2[i] = new TCanvas(Form("cTOR%i",i),"",400,400);
    c2[i]->cd();
    h2[i]->Draw("phist c");
    c2[i]->SaveAs(Form("cRFQ_%s.png",sTOR[i].c_str()));
  }

}
