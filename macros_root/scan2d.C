void scan2d(std::string filename) {
 
  TFile *f = new TFile("RFQ_B.root","RECREATE");
  //TH1F *h1 = new TH1F("h1","x distribution",100,-4,4);
  TTree *T = new TTree("paramT","RFQ vs B");
  T->ReadFile(filename.c_str(),"RFQPAH_S/D:RFBPAH_S/D:RFQPAH_R/D:RFBPAH_R/D:TO1IN_R/D:TO3IN_R/D:D7TOR_R/D");
  double nlines = T->GetEntries();
  printf(" found %f points\n",nlines);
  //T->Draw("RFQPAH_S:RFBPAH_S");
  T->Write();
  f->Close();
}
