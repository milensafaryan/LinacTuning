using namespace std;
using namespace ROOT;

double LM[7][4];
string sLM[7][4];
double RFQp[2];
double Bp[2];
double T5p[2];
double delta = 0.1;
double mid[3] = {159.,236.,-30.}; //RFQ, B, T5

TH1D* h1[7][4]; //vs RFQ
TH1D* h2[7][4]; //vs B
TH1D* h3[7][4]; //vs T5

double bins[9] ={215.5,221.,226.5,231.5,237.,242.,247.5,252.5,258.};

void plot_losses_3phase(string name="3phase"){

  TFile* fin = new TFile(Form("%s.root",name.c_str()));
  TTree* T = (TTree*)fin->Get("paramT");
  
  /* Set branches */
  T->SetBranchAddress("RFQPAH_S",&RFQp[0]);
  T->SetBranchAddress("RFQPAH_R",&RFQp[1]);
  T->SetBranchAddress("RFBPAH_S",&Bp[0]);
  T->SetBranchAddress("RFBPAH_R",&Bp[1]);
  T->SetBranchAddress("V5QSET_S",&T5p[0]);
  T->SetBranchAddress("V5QSET_R",&T5p[1]);
  for(int i=0; i<7; i++){
    for(int j=0; j<4; j++){
      sLM[i][j] = Form("D%i%iLM_R",i+1,j+1);
      //cout << sLM[i][j] << endl;
      h1[i][j] = new TH1D(Form("hRFQ_D%i%i",i+1,j+1),Form(";RFQ phase [deg];D%i%iLM (normalized)",i+1,j+1),9,128.,191.);
      h2[i][j] = new TH1D(Form("hB_D%i%i",i+1,j+1),Form(";Buncher phase [deg];D%i%iLM (normalized)",i+1,j+1),8,bins);
      h3[i][j] = new TH1D(Form("hT5_D%i%i",i+1,j+1),Form(";Tank5 phase [deg];D%i%iLM (normalized)",i+1,j+1),6,-33.,-28.5);
      T->SetBranchAddress(sLM[i][j].c_str(),&LM[i][j]);
    }
  }

  /* Loop over entries */
  for(int k=0; k<T->GetEntries(); k++){
    T->GetEntry(k);
    
    if( Bp[0]>mid[1]-delta && Bp[0]<mid[1]+delta && T5p[0]>mid[2]-delta && T5p[0]<mid[2]+delta){
      for(int i=0; i<7; i++){
	for(int j=0; j<4; j++){
	  h1[i][j]->Fill(RFQp[1],LM[i][j]);
	}
      }
    }
    
    if( RFQp[0]>mid[0]-delta && RFQp[0]<mid[0]+delta && T5p[0]>mid[2]-delta && T5p[0]<mid[2]+delta){
      //cout << "set: " <<Bp[0] << " read: "<<Bp[1] << endl;
      for(int i=0; i<7; i++){
	for(int j=0; j<4; j++){
	  h2[i][j]->Fill(Bp[1],LM[i][j]);
	}
      }
    }    
    if( RFQp[0]>mid[0]-delta && RFQp[0]<mid[0]+delta && Bp[0]>mid[1]-delta && Bp[0]<mid[1]+delta){
      //cout << "set: " <<T5p[0] << " read: "<<T5p[1] << endl;
      for(int i=0; i<7; i++){
	for(int j=0; j<4; j++){
	  h3[i][j]->Fill(T5p[1],LM[i][j]);
	}
      }
    }
    
  }

  /* Normalize histograms 
  for(int i=0; i<7; i++){
    for(int j=0; j<4; j++){
      h1[i][j]->Scale(1./h1[i][j]->Integral());
      h2[i][j]->Scale(1./h2[i][j]->Integral());
      h3[i][j]->Scale(1./h3[i][j]->Integral());
    }
  }
  */
  //for(int b=0; b<6; b++) cout << h3[6][3]->GetBinContent(b+1) << endl;
  TCanvas* c1[7][4]; 
  for(int i=0; i<7; i++){
    for(int j=0; j<4; j++){
      c1[i][j]  = new TCanvas(Form("c%i%i",i+1,j+1),"",900,300);
      c1[i][j]->Divide(3,1);
      c1[i][j]->cd(1);
      h1[i][j]->Draw();
      c1[i][j]->cd(2);
      h2[i][j]->Draw();
      c1[i][j]->cd(3);
      h3[i][j]->Draw();
      c1[i][j]->SaveAs(Form("c%i%i.png",i+1,j+1));
    }
  }

}
