#include "TGraph.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TAxis.h"
#include "TGaxis.h"
#include "TLegend.h"

void plot_tgraphs(std::string file="tank5", std::string param="V5QSET")
{
  gStyle->SetPadTickY(0);
  gStyle->SetOptStat(0);
  // note: this macro draws graphs with different x- and y- axes ranges

  file +=".root";
  TFile* f1 = new TFile(file.c_str());
  TGraph *g1 = (TGraph*)f1->Get("grD7TOR");
  g1->SetTitle(Form("D7TOR;%s [deg];Beam current",param.c_str()));
  g1->SetMarkerStyle(20);
  g1->SetMarkerColor(1);
  g1->SetLineColor(1);
  g1->GetHistogram()->SetMaximum(24.6);
  g1->GetHistogram()->SetMinimum(22.6);
  
  TGraph *g2 = (TGraph*)f1->Get("grTO5OUT");
  g2->SetTitle("TO5OUT;;");
  g2->SetMarkerStyle(20);
  g2->SetMarkerColor(4);
  g2->SetLineColor(4);

  TGraph *g5 = (TGraph*)f1->Get("grTO1IN");
  g5->SetTitle("TO1IN;;");
  g5->SetMarkerStyle(20);
  g5->SetMarkerColor(6);
  g5->SetLineColor(6);

  TGraph *g6 = (TGraph*)f1->Get("grTO3IN");
  g6->SetTitle("TO3IN;;");
  g6->SetMarkerStyle(20);
  g6->SetMarkerColor(8);
  g6->SetLineColor(8);

  TGraph *g3 = (TGraph*)f1->Get("grD7LMSM");
  g3->SetTitle("D7LMSM;;Summed beam loss");
  g3->SetMarkerStyle(20);
  g3->SetMarkerColor(2);
  g3->SetLineColor(2);
  g3->GetHistogram()->SetMaximum(10.5);
  g3->GetHistogram()->SetMinimum(3.5);

  // Subtract g1 from g2
  TGraph* g4 = new TGraph(g1->GetN());
  g4->SetTitle(Form("TO5OUT-D7TOR;%s [deg];#Delta Beam current",param.c_str()));
  g4->SetMarkerStyle(20);
  for(int i=0; i<g1->GetN(); i++){
    g4->SetPoint(i, g2->GetPointX(i),g2->GetPointY(i)-g1->GetPointY(i));
  }
  g4->GetHistogram()->SetMaximum(0.50);
  g4->GetHistogram()->SetMinimum(0.18);

  /* New axes and legends */
  Double_t xmin = g1->GetHistogram()->GetXaxis()->GetXmin();
  Double_t xmax = g1->GetHistogram()->GetXaxis()->GetXmax();
  Double_t dx = (xmax - xmin) / 0.8; // 10 percent margins left and right

  Double_t ymin = g3->GetHistogram()->GetMinimum();
  Double_t ymax = g3->GetHistogram()->GetMaximum();
  Double_t dy = (ymax - ymin) / 0.8; // 10 percent margins top and bottom
  /*
  TGaxis *xaxis = new TGaxis(xmin, ymax, xmax, ymax, xmin, xmax, 510, "-L");
  xaxis->SetLineColor(kRed);
  xaxis->SetLabelColor(kRed);
  xaxis->Draw();
  gPad->Update();
  */
  TGaxis *yaxis = new TGaxis(xmax, ymin, xmax, ymax, ymin, ymax, 510, "+L");
  yaxis->SetLineColor(kRed);
  yaxis->SetLabelColor(kRed);
  yaxis->SetTitle("Summed beam loss");
  yaxis->SetTitleColor(kRed);

  TLegend *leg = new TLegend(0.66, 0.25, 0.86, 0.40);
  leg->SetFillColor(0);
  leg->SetTextSize(0.036);
  leg->SetBorderSize(0);
  leg->AddEntry(g1, "D7TOR", "L");
  leg->AddEntry(g5, "TO1IN", "L");
  leg->AddEntry(g6, "TO3IN", "L");
  leg->AddEntry(g2, "TO5OUT", "L");
  leg->AddEntry(g3, "D7LMSM", "L");

  TLegend *leg2 = new TLegend(0.66, 0.25, 0.86, 0.40);
  leg2->SetFillColor(0);
  leg2->SetTextSize(0.036);
  leg2->SetBorderSize(0);
  leg2->AddEntry(g4, "TO5OUT-D7TOR", "L");
  leg2->AddEntry(g3, "D7LMSM", "L");

  /* Canvas #1 */
  TCanvas *c = new TCanvas("c", "Beam current", 200, 10, 700, 500);
  c->cd();
  TPad *p1 = new TPad("p1", "", 0, 0, 1, 1);
  p1->SetGrid();
  TPad *p2 = new TPad("p2", "", 0, 0, 1, 1);
  p2->SetFillStyle(4000); // will be transparent

  p1->Draw();
  p1->cd();
  
  g1->Draw("ALP");
  g2->Draw("LPSAME");
  g5->Draw("LPSAME");
  g6->Draw("LPSAME");
  gPad->Update();
  
  p2->Range(xmin-0.1*dx, ymin-0.1*dy, xmax+0.1*dx, ymax+0.1*dy);
  p2->Draw();
  p2->cd();
  g3->Draw("LP");
  gPad->Update();

  yaxis->Draw();
  gPad->Update();
  // p1->cd();

  leg->Draw();
  gPad->Update();
  /* End of canvas #1 */

  /* Canvas #2 */
  TCanvas *c3 = new TCanvas("c3", "Delta beam current", 200, 10, 700, 500);
  c3->cd();
  TPad *p3 = new TPad("p3", "", 0, 0, 1, 1);
  p3->SetGrid();
  TPad *p4 = new TPad("p4", "", 0, 0, 1, 1);
  p4->SetFillStyle(4000); // will be transparent

  p3->Draw();
  p3->cd();
  g4->Draw("APL");
  gPad->Update();
  p4->Range(xmin-0.1*dx, ymin-0.1*dy, xmax+0.1*dx, ymax+0.1*dy);
  p4->Draw();
  p4->cd();
  g3->Draw("LP");
  gPad->Update();

  yaxis->Draw();
  gPad->Update();

  leg2->Draw();
  gPad->Update();
  /* End of canvas #2 */

  /* Canvas #3 */
  TH2D* h2 = (TH2D*)f1->Get("hLoss");
  h2->SetMinimum(-0.1);
  h2->SetMaximum(7.50);
  TCanvas* c2 = new TCanvas("c2", "", 200, 10, 700, 500);
  c2->cd();
  h2->Draw("colz");
  /* End of canvas #3 */

}
