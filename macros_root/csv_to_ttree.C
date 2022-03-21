
void csv_to_ttree(std::string name="tank5", std::string par1="V5QSET",std::string par2="", std::string par3="", int n_step=13, int n_sample=20){

  gStyle->SetPalette(56);

  auto fileName = Form("/Users/ralitsa/linacTune/devicescan_%s.csv",name.c_str());
  auto tdf = ROOT::RDF::MakeCsvDataFrame(fileName);

  std::vector<std::string> colNames = tdf.GetColumnNames();
  std::string query = "";
  std::vector<std::string> filteredCols;
  filteredCols.push_back("step");
  filteredCols.push_back(Form("%s_S",par1.c_str()));
  filteredCols.push_back(Form("%s_R",par1.c_str()));
  if(!par2.empty()){
    filteredCols.push_back(Form("%s_S",par2.c_str()));
    filteredCols.push_back(Form("%s_R",par2.c_str()));
  }
  if(!par3.empty()){
    filteredCols.push_back(Form("%s_S",par3.c_str()));
    filteredCols.push_back(Form("%s_R",par3.c_str()));
  }

  int LMcnt=0;
  int TORcnt=0;
  for(int i=0; i<colNames.size(); i++){
    
    if( colNames[i].find("Timestamp")==std::string::npos && colNames[i].find("LM")!=std::string::npos){
      query += colNames[i]+"+";
      filteredCols.push_back(colNames[i]);
      LMcnt++;
      //std::cout << colNames[i] << std::endl;
    }
    
    if( colNames[i].find("Timestamp")==std::string::npos && colNames[i].find("TO")!=std::string::npos){
      filteredCols.push_back(colNames[i]);
      TORcnt++;
      //std::cout << colNames[i] << std::endl;
    }
  }
  query.pop_back();
  filteredCols.push_back("LMSM");

  std::cout <<"Number of loss monitors: "<< LMcnt <<" Number of toroids: "<< TORcnt << std::endl;

  auto filteredEvents = tdf.Filter("iteration == 0").Define("LMSM", query.c_str());
  filteredEvents.Snapshot("paramT",Form("%s.root",name.c_str()),filteredCols);
  std::cout << "Created snapshot " << name << std::endl;

  /*
  TFile* f1 = new TFile(Form("%s.root",name.c_str()),"UPDATE");
  TTree* T2 = new TTree("avgT","averaged");
  TTree* T = (TTree*)f1->Get("paramT");
  double min1 = T->GetMinimum(Form("%s_R",par1.c_str()));
  double max1 = T->GetMaximum(Form("%s_R",par1.c_str()));

  double means[filteredCols.size()];
  std::vector<std::string> refilteredCols;
  refilteredCols.push_back("step");

  for(int k=0; k<filteredCols.size(); k++){
    if( filteredCols[k].find(par1)==std::string::npos 
	&& filteredCols[k].find(par2)==std::string::npos 
	&& filteredCols[k].find(par3)==std::string::npos 
	&& filteredCols[k].find("TO")==std::string::npos 
	&& filteredCols[k].find("LM")==std::string::npos) continue;
    refilteredCols.push_back(filteredCols[k]);
    T2->Branch(Form("%s",filteredCols[k].c_str()), &means[k],Form("%s/D",filteredCols[k].c_str()));
  }
  */
  /*
  TH2D* hLoss = new TH2D("hLoss",Form(";;%s (deg)",param.c_str()), LMcnt, 0, LMcnt, n_step,min1,max1);
  for (int b=0; b<LMcnt; b++){ hLoss->GetXaxis()->SetBinLabel(b+1,(filteredCols[b+3+TORcnt].substr(0,filteredCols[b+3+TORcnt].length()-2)).c_str());}
  hLoss->GetXaxis()->LabelsOption("v");
  TGraph* grTor1 = new TGraph(n_step);
  TGraph* grTor2 = new TGraph(n_step);
  TGraph* grTor3 = new TGraph(n_step);
  TGraph* grTor4 = new TGraph(n_step);
  TGraph* grL1 = new TGraph(n_step);

  int k1=0;
  int k2=0;
  int k3=0;
  int k4=0;
  int k5=0;
  
  std::cout << "Making graph of D7TOR vs. " << param << std::endl;
  std::cout << "Making graph of TO5OUT vs. " << param << std::endl;
  std::cout << "Making graph of TO1IN vs. " << param << std::endl;
  std::cout << "Making graph of TO3IN vs. " << param << std::endl;
  std::cout << "Making graph of D7LMSM vs. " << param << std::endl;
  std::cout << "Making 2D histogram of losses vs. "<< param << std::endl;
  */
  /*
  for(int i=0; i<n_step; i++){
    auto dr = filteredEvents.Range(i*n_sample, (i+1)*n_sample);      
    for(int j=1; j<refilteredCols.size();j++){
    if( filteredCols[j].find(par1)==std::string::npos 
	&& filteredCols[j].find(par2)==std::string::npos 
	&& filteredCols[j].find(par3)==std::string::npos 
	&& filteredCols[j].find("TO")==std::string::npos 
	&& filteredCols[j].find("LM")==std::string::npos) continue;
      means[j] = *dr.Mean(filteredCols[j]);
      T2->Fill();
      
      if( filteredCols[j].find("D7TOR")!=std::string::npos) k1=j;
      else if( filteredCols[j].find("TO5OUT")!=std::string::npos) k2=j;
      else if( filteredCols[j]. find("TO1IN")!=std::string::npos) k3=j;
      else if( filteredCols[j]. find("TO3IN")!=std::string::npos) k4=j;
      else if( filteredCols[j].find("LM")!=std::string::npos){
	if( filteredCols[j].find("LMSM")!=std::string::npos) k5=j;
	else hLoss->SetBinContent(j-3-TORcnt+1,i+1,means[j]);
      } 
      
      //std::cout << i <<" "<< filteredCols[j] <<" "<< means[j] << std::endl;
    }
    //grTor1->SetPoint(i, means[2], means[k1]);
    //grTor2->SetPoint(i, means[2], means[k2]);
    //grTor3->SetPoint(i, means[2], means[k3]);
    //grTor4->SetPoint(i, means[2], means[k4]);
    //grL1->SetPoint(i, means[2], means[k5]);
  }
  
  grTor1->Write("grD7TOR");
  grTor2->Write("grTO5OUT");
  grTor3->Write("grTO1IN");
  grTor4->Write("grTO3IN");
  grL1->Write("grD7LMSM");
  hLoss->Write();
  */
  //f1->cd();
  //T2->Write();
  //f1->Close();
  std::cout << "All done!" << std::endl;
}
