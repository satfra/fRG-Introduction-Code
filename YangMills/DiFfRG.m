(* ::Package:: *)

(************************************************************************)
(* This file was generated automatically by the Mathematica front end.  *)
(* It contains Initialization cells from a Notebook file, which         *)
(* typically will have the same name as this file except ending in      *)
(* ".nb" instead of ".m".                                               *)
(*                                                                      *)
(* This file is intended to be loaded into the Mathematica kernel using *)
(* the package loading commands Get or Needs.  Doing so is equivalent   *)
(* to using the Evaluate Initialization Cells menu command in the front *)
(* end.                                                                 *)
(*                                                                      *)
(* DO NOT EDIT THIS FILE.  This entire file is regenerated              *)
(* automatically each time the parent Notebook file is saved in the     *)
(* Mathematica front end.  Any changes you make to this file will be    *)
(* overwritten.                                                         *)
(************************************************************************)



(* ::Input::Initialization:: *)
Needs["FormTracer`"]
Needs["QMeSderivation`"]
Get["FeynmanGraphs.m"]

$DistributedContexts:={$Context,"FormTracer`"}


(* ::Input::Initialization:: *)
DefineLorentzTensors[deltaLorentz[mu, nu], vec[p, mu], sp[p, q], eps[i, j, k], deltaSpinor[i, j], gamma[mu, i, j], gamma5[i, j], vecs[p, mu], sps[p, q]];
DefineFormAutoDeclareFunctions[{sigma}];
AddExtraVars[\[Pi], k, E, I];
DefineGroupTensors[{
   {SUNfund, {color, Nc}, deltaadjCol[a, b], SUNFCol[a, b, c], deltaFundCol[a, b], SUNTCol[a, l, j], adjEpsCol[a, b, c], epsCol[a, b, c]},
   {SUNfund, {flavor, Nf}, deltaMesonFlav[a, b], SUNFFlav[a, b, c], deltaFundFlav[a, b], SUNTFlav[a, l, j], adjEpsFlav[a, b, c], epsFlav[a, b, c]}
   }];
SUNTFlav[0, f1_, f2_] := deltaFundFlav[f1, f2]/Sqrt[2 Nf]

FiniteT[False];
DisentangleLorentzStructures[False];
PartialTrace[False];


(* ::Input::Initialization:: *)
GetDirectory[]:=If[$Notebooks,NotebookDirectory[],Directory[]]<>"/";

AutoExport[]:=SetOptions[EvaluationNotebook[],{AutoGeneratedPackage->Automatic,InitializationCellEvaluation->False,InitializationCellWarning->False,StyleDefinitions->Notebook[{Cell[StyleData[StyleDefinitions->"Default.nb"]],Cell[StyleData["Input"],InitializationCell->True]},Visible->False,StyleDefinitions->"PrivateStylesheetFormatting.nb"]}];


AutoSaveRestore[fileName_String,expr_]:=Module[{ret},
If[FileExistsQ[fileName<>".m"],
ret=Import[fileName<>".m"];
Print["Imported existing file \""<>fileName<>".m"<>"\"..."];
,
ret=ReleaseHold[expr];
Export[fileName<>".m",ret];
Print["Saved to file \""<>fileName<>".m"<>"\"..."];
];
ret
];
SetAttributes[AutoSaveRestore,HoldRest]


(* ::Input::Initialization::Plain:: *)
pdash[p_,i_,j_]:=Module[{mu},gamma[mu,i,j]vec[p,mu]];
psdash[p_,i_,j_]:=Module[{mu},gamma[mu,i,j]vecs[p,mu]];

sigma[v1_,v2_,d1_,d2_]:=Module[{dint1,dint2},I/2 (gamma[v1,d1,dint1]gamma[v2,dint1,d2]-gamma[v2,d1,dint2]gamma[v1,dint2,d2])];

SetAttributes[sps,Orderless]
SetAttributes[sp,Orderless]
SetAttributes[cos,Orderless]

sps[0,a_]=0;
sps[a_,0]=0;
sps[0,0]=0;
sp[0,a_]=0;
sp[a_,0]=0;
sp[0,0]=0;

vec[0,mu_]=0
vecf[0,mu_]=0
vecs[0,mu_]=0
vecs[p_,0]=0

sps[a_,b_+c_]:= sps[a,b]+sps[a,c];
sps[a_,-1b_]:= -sps[a,b];
sps[b_,a_?NumericQ c_]:= a sps[b,c];

sp[a_,b_+c_]:= sp[a,b]+sp[a,c];
sp[a_,-1b_]:= -sp[a,b];
sp[b_,a_?NumericQ c_]:= a sp[b,c];

vecs[p_+q_,mu_]:=vecs[p,mu]+vecs[q,mu];
vecs[-1 p_,mu_]:= -vecs[p,mu];
vecs[n_?NumericQ a_,mu_]:=n vecs[a,mu];

vec[p_+q_,mu_]:=vec[p,mu]+vec[q,mu];
vec[-1 p_,mu_]:= -vec[p,mu];
vec[n_?NumericQ a_,mu_]:=n vec[a,mu];


GetAllSymbols[expr_]:=DeleteDuplicates@Cases[{expr},_Symbol,Infinity]
ReduceDiagramToMomenta[diag_,momenta_]:=diag/.Cor_[{k___}]:>Cor@@Select[{k},ContainsAny[GetAllSymbols[#],momenta]&]


ProjectDiagramsToSymmetricPoint[diagramsIn_,setup_,q_,moms___]:=Module[{diagrams,momenta,multiplicities,bosonicPropsOld,bosonicProps,bosonicPropsRules,momentumConservation,momentumConservationFull,reducedDiagrams,perms,permRules,PReplacement,possibleExpr,curExpr,i,j},
diagrams=diagramsIn;
momenta={moms};
multiplicities=Map[1&,diagrams];

bosonicPropsOld=Map[Symbol["G"<>ToString[Head[#]]<>ToString[Head[#]]]&,setup["FieldSpace"]["bosonic"]];
bosonicProps=Unique[bosonicPropsOld];
bosonicPropsRules=Thread[bosonicPropsOld->bosonicProps];
Map[SetAttributes[#,Orderless]&,bosonicProps];
diagrams=diagrams//.bosonicPropsRules;

momentumConservation={Total[-momenta]->0,Total[momenta]->0};
momentumConservationFull=momenta[[-1]]->-Total[momenta[[1;;-2]]];

reducedDiagrams=ReduceDiagramToMomenta[diagrams,{q,moms}]//.momentumConservation;
perms=Map[Permutations[#]&,Select[Subsets[momenta],Length[#]>=2&]];
permRules=Flatten[Map[Table[Thread[#[[1]]->#[[i]]],{i,1,Length[#]}]&,perms],1];
PReplacement=Map[#->-#&,momenta];

For[i=1,i<=Length[diagrams],i++,
For[j=i+1,j<=Length[diagrams],j++,
If[multiplicities[[i]]!=0&&multiplicities[[j]]!=0,
possibleExpr=Map[reducedDiagrams[[j]]/.#&,permRules]//.momentumConservationFull//Simplify;
curExpr=reducedDiagrams[[i]]//.momentumConservationFull//Simplify;
If[AnyTrue[possibleExpr,curExpr===#&],
multiplicities[[i]]+=1;
multiplicities[[j]]-=1;,
If[AnyTrue[possibleExpr/.PReplacement,curExpr===#&],
multiplicities[[i]]+=1;
multiplicities[[j]]-=1;
]
];
]
]
];

If[Total[multiplicities]!=Length[diagrams],Print["Something went wrong in identifying same diagrams!"];Abort[]];
Map[ClearAll[#]&,bosonicProps];
Select[multiplicities diagramsIn,#=!={0}&]
]


separateScalarProducts3D[expr_]:=Module[{},expr//.{sp[q_,p_]:>sps[q,p]+vec[q,0]vec[p,0],vec[p_,mu_/;mu=!=0]:>vecs[p,mu]+deltaLorentz[mu,0]vec[p,0]}]
expandScalarProducts3D[expr_]:=Module[{q,p},expr//.{sp[q_,p_]->sps[q,p]+vec[q,0]vec[p,0]}//.{sps[q_,p_]->q p cos[q,p], cos[p_,p_]->1}]
expandScalarProducts4D[expr_]:=Module[{q,p},expr//.{sp[q_,p_]->q p cos[q,p], cos[p_,p_]->1}]
Derivative[0,1][cos][p_,q_]=0;
Derivative[1,0][cos][p_,q_]=0;
Derivative[1,0][vec][p_,q_]=0;
cos[p_,p_]=1;

simplifyMomenta3D[q_Symbol,p_Symbol,expr_]:=expr//.{
cos[p,q]:>Symbol["cos1"],
cos[p,Symbol[ToString[q]<>"f"]]:>Symbol["cos1"],
vec[p,0]:>Symbol[ToString[p]<>"0"],
vec[q,0]:>Symbol[ToString[q]<>"0"],
vec[Symbol[ToString[q]<>"f"],0]:>Symbol[ToString[q]<>"0"]+\[Pi] T
}//.{
Symbol[ToString[q]<>"f"]->q
}

simplifyMomenta4D[q_Symbol,p_Symbol,expr_]:=expr//.{
cos[p,q]:>Symbol["cos1"],
cos[p,Symbol[ToString[q]<>"f"]]:>Symbol["cos1"]
}//.{
Symbol[ToString[q]<>"f"]->q
}


MatsubaraSum[expr_,p0_Symbol]:=Module[{denom,poles,residues,n,r,SumB},
denom = Denominator[expr];
poles = p0/.{ToRules[Reduce[denom==0,p0]]};
residues={};
For[n=1,n<=Length[poles],n++,
r=Simplify[Residue[expr Coth[(I p0)/(2T)],{p0,poles[[n]]}]];
AppendTo[residues,r];];
SumB=1/(2I) residues;
Total[SumB]
];


ProjectToSymmetricPoint4D[expr_,q_Symbol,p_Symbol,momenta___Symbol]:=Module[{momentaList,nMomenta,rules,qf},
momentaList={momenta};
nMomenta=Length[momentaList];
qf=Symbol[ToString[q]<>"f"];
rules=Map[sp[#[[1]],#[[2]]]->-(1/(nMomenta-1))sp[p,p]&,Subsets[momentaList,{2}]]
\[Union]Map[sp[#,#]->sp[p,p]&,momentaList]
\[Union]Map[sp[#,q]->Symbol["cos"~~ToString[#]] p q&,momentaList[[1;;nMomenta-1]]]
\[Union]Map[sp[#,qf]->Symbol["cos"~~ToString[#]] p qf&,momentaList[[1;;nMomenta-1]]]
\[Union]{momentaList[[nMomenta]]->-Total[momentaList[[1;;nMomenta-1]]]};
expr//.rules
]


$StandardSimplify=Simplify;
SetStandardSimplify[sim_]:=Module[{},
$StandardSimplify=sim;
]


AdjustToNKernels[nKernels_Integer]:=Module[{},
If[nKernels!=0,
CloseKernels[];
LaunchKernels[nKernels];
DistributeDefinitions["FormTracer`"];
DistributeDefinitions[ExpandedFormTrace,InsertChargeConjRules]
];
]
RestoreKernels[]:=Module[{},
CloseKernels[];
LaunchKernels[];
DistributeDefinitions["FormTracer`"];
DistributeDefinitions[ExpandedFormTrace,InsertChargeConjRules];
]
DiFfRGMap[nKernels_Integer]:=Module[{},
If[nKernels<=1,
Return[ResourceFunction["DynamicMap"]],
Return[ResourceFunction["DynamicMap"][#1,#2,Parallel->nKernels]&]
];
];


(* ::Input::Initialization:: *)
SimpAll[ex_]:=Module[{},
DiFfRGMap[2][$StandardSimplify, ex]
  ];
DoubleSimp[expr_] := Quiet[Simplify[Simplify[expr, TimeConstraint -> 0.1]]];
QuickSimp[expr_] := Simplify[expr,TimeConstraint -> 0.1] // Quiet;


ExpandedFormTrace[expr_,disentangle_/;BooleanQ[disentangle]]:=Module[{},
Block[{Print},DisentangleLorentzStructures[disentangle]];
FormTrace[expr]
];


IterativelySum[expr_List,nKernels_Integer]:=Module[{returnValue},
returnValue=expr;
If[Length[returnValue]==1,Return[returnValue]];
returnValue=DiFfRGMap[nKernels][
QuickSimp[Total[#]]&
,Partition[returnValue,UpTo[Ceiling[Length[returnValue]/16]]]
];
returnValue=DiFfRGMap[nKernels][
QuickSimp[Total[#]]&
,Partition[returnValue,UpTo[Ceiling[Length[returnValue]/8]]]
];
returnValue=DiFfRGMap[nKernels][
QuickSimp[Total[#]]&
,Partition[returnValue,UpTo[Ceiling[Length[returnValue]/4]]]
];
returnValue=DiFfRGMap[nKernels][
QuickSimp[Total[#]]&
,Partition[returnValue,UpTo[Ceiling[Length[returnValue]/2]]]
];
returnValue
]


(* ::Input::Initialization:: *)
TraceToSP4D[nKernels_Integer,postfix_String,ex_,q_Symbol,p_Symbol,momenta___Symbol]:=TraceToSP4D[$StandardSimplify,nKernels,postfix,ex,q,p,momenta];
TraceToSP4D[simpFunc_,nKernels_Integer,postfix_String,ex_,q_Symbol,p_Symbol,momenta___Symbol]:=Module[{returnValue,files,tempDir},

tempDir=GetDirectory[]<>"TraceBuffer/";

If[FileExistsQ[tempDir<>postfix<>"/result.m"],
Print["Using result \""<>tempDir<>postfix<>"/result.m"<>"\" from buffer..."];
Return[Import[tempDir<>postfix<>"/result.m"]];
];

Print["Tracing..."];
AdjustToNKernels[nKernels];
If[Not@DirectoryQ[tempDir<>postfix],CreateDirectory[tempDir<>postfix]];

DisentangleLorentzStructures[True];
DiFfRGMap[nKernels][
(If[FileExistsQ[tempDir<>postfix<>"/tmp"<>ToString[#]<>".m"],
tempDir<>postfix<>"/tmp"<>ToString[#]<>".m",
Export[tempDir<>postfix<>"/tmp"<>ToString[#]<>".m",
simpFunc[
simplifyMomenta4D[q,p,
expandScalarProducts4D[
ProjectToSymmetricPoint4D[
ExpandedFormTrace[ProjectToSymmetricPoint4D[ex[[#]],q,p,momenta],True]
,q,p,momenta]
]
]
]
]
]
)&
,Table[i,{i,1,Length[ex]}]
];

Print["...done. Summing subterms..."];
files=Map[tempDir<>postfix<>"/tmp"<>ToString[#]<>".m"&,Table[i,{i,1,Length[ex]}]];
returnValue=Map[
Import[#]&
,files
];
returnValue=IterativelySum[returnValue,nKernels];
RestoreKernels[];

returnValue=simpFunc[Total[returnValue]];
Print["...done."];
Export[tempDir<>postfix<>"/result.m",returnValue];
returnValue
]


(* ::Input::Initialization:: *)
TracePropagator4D[nKernels_Integer,postfix_String,ex_,q_Symbol,p_Symbol]:=TracePropagator4D[$StandardSimplify,nKernels,postfix,ex,q,p]
TracePropagator4D[simpFunc_,nKernels_Integer,postfix_String,ex_,q_Symbol,p_Symbol]:=Module[{returnValue,files,tempDir},
tempDir=GetDirectory[]<>"TraceBuffer/";

If[FileExistsQ[tempDir<>postfix<>"/result.m"],
Print["Using result \""<>tempDir<>postfix<>"/result.m"<>"\" from buffer..."];
Return[Import[tempDir<>postfix<>"/result.m"]];
];

Print["Tracing..."];
AdjustToNKernels[nKernels];
If[Not@DirectoryQ[tempDir<>postfix],CreateDirectory[tempDir<>postfix]];

DiFfRGMap[nKernels][
(If[Not@FileExistsQ[tempDir<>postfix<>"/tmp"<>ToString[#]<>".m"],
Export[tempDir<>postfix<>"/tmp"<>ToString[#]<>".m",
simpFunc[
simplifyMomenta4D[q,p,
expandScalarProducts4D[
ExpandedFormTrace[ex[[#]],False]
]
]
]
]
]
)&
,Table[i,{i,1,Length[ex]}]
];

Print["...done. Summing subterms..."];
files=Map[tempDir<>postfix<>"/tmp"<>ToString[#]<>".m"&,Table[i,{i,1,Length[ex]}]];
returnValue=Map[
Import[#]&
,files
];
returnValue=IterativelySum[returnValue,nKernels];
RestoreKernels[];

returnValue=simpFunc[Total[returnValue]];
Print["...done."];
Export[tempDir<>postfix<>"/result.m",returnValue];
returnValue
]


(* ::Input::Initialization:: *)
TraceToSG4D[nKernels_Integer,postfix_String,ex_,q_Symbol,p_Symbol,pA_Symbol,pqb_Symbol,pq_Symbol]:=TraceToSG4D[$StandardSimplify,nKernels,postfix,ex,q,p,pA,pqb,pq];
TraceToSG4D[simpFunc_,nKernels_Integer,postfix_String,ex_,q_Symbol,p_Symbol,pA_Symbol,pqb_Symbol,pq_Symbol]:=Module[{returnValue,files,tempDir},

tempDir=GetDirectory[]<>"TraceBuffer/";

If[FileExistsQ[tempDir<>postfix<>"/result.m"],
Print["Using result \""<>tempDir<>postfix<>"/result.m"<>"\" from buffer..."];
Return[Import[tempDir<>postfix<>"/result.m"]];
];

Print["Tracing..."];
AdjustToNKernels[nKernels];
If[Not@DirectoryQ[tempDir<>postfix],CreateDirectory[tempDir<>postfix]];

DisentangleLorentzStructures[True];
DiFfRGMap[nKernels][
(If[FileExistsQ[tempDir<>postfix<>"/tmp"<>ToString[#]<>".m"],
tempDir<>postfix<>"/tmp"<>ToString[#]<>".m",
Export[tempDir<>postfix<>"/tmp"<>ToString[#]<>".m",
simpFunc[
simplifyMomenta4D[q,p,
expandScalarProducts4D[ExpandedFormTrace[ex[[#]]//.{pA->0,pqb->-p,pq->p},True]
]
]
]
]
]
)&
,Table[i,{i,1,Length[ex]}]
];

Print["...done. Summing subterms..."];
files=Map[tempDir<>postfix<>"/tmp"<>ToString[#]<>".m"&,Table[i,{i,1,Length[ex]}]];
returnValue=Map[
Import[#]&
,files
];
returnValue=IterativelySum[returnValue,nKernels];
RestoreKernels[];

returnValue=simpFunc[Total[returnValue]];
Print["...done."];
Export[tempDir<>postfix<>"/result.m",returnValue];
returnValue
]


(* ::Input::Initialization:: *)
Print["Loaded DiFfRG Mathematica package
Author: Franz Richard Sattler (sattler@thphys.uni-heidelberg.de)
Version: 0.1
Year: 2024
"]
