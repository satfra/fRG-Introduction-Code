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



BeginPackage["DiFfRG`FeynmanGraphs`"];


PlotDiagrams::usage = "Plot all diagrams from a set of derivative rules and a fRG setup";


DrawOneDiagram::usage="Plot one diagram from a set of derivative rules and a superindex diagram";


Begin["Private`"];


(* ::Input::Initialization:: *)
DrawOneDiagram[diagram_, derivativeList_, styleOpts_] :=
    Module[
        {pick, cross, externalIndices, idx,mvertexName,vertexName, edges,element, edgesType, vertices,
             merges, subidx, propagators, styledPropagators, prefactor, style, edgeStyle,
             vertexShapes, vertexLabels}
        ,
        (*convenience function for later*)
        pick[list_, obj_] := list[[Position[list, obj][[1]][[1]]]] /.
             a_[b_] -> a;
        cross[r_] := Graphics[{Thick, Line[{{r / Sqrt[2], r / Sqrt[2]
            }, {-r / Sqrt[2], -r / Sqrt[2]}}], Line[{{r / Sqrt[2], -r / Sqrt[2]},
             {-r / Sqrt[2], r / Sqrt[2]}}], Circle[{0, 0}, r]}];
        (*get the external indices*)
        externalIndices = {};
        Clear[idx];
        For[idx = 1, idx <= Length[derivativeList], idx++,
            AppendTo[externalIndices, Evaluate @@ derivativeList[[idx
                ]]];
        ];
        Clear[idx];
        (*extract all elements from the diagram*)
        edges = {};
        edgesType = {};
        vertices = {};
        merges = {};
        Clear[idx];
        For[idx = 1, idx <= Length[diagram], idx++,
            element = diagram[[idx]];

            (*In case it is a propagator, we have a new edge*)
            If[element["type"] == "Propagator",
                AppendTo[edges, element["indices"][[1]][[2]][[1]] \[UndirectedEdge] element[
                    "indices"][[2]][[2]][[1]]];
                AppendTo[edgesType, ToString[element["indices"][[1]][[
                    1]]] <> ToString[element["indices"][[2]][[1]]]]
                ,
                0
            ];

            (*In case it is a nPoint, we have rules for merging vertices*)
            If[element["type"] == "nPoint",
                mvertexName = "";
                For[subidx = 1, subidx <= element["nPoint"], subidx++,
                    
                    vertexName = mvertexName <> ToString[element["indices"
                        ][[subidx]][[1]]]
                ];
                vertexName = mvertexName <> ToString[Length[vertices]
                    ];
                AppendTo[vertices, vertexName];
                Clear[subidx];
                For[subidx = 1, subidx <= element["nPoint"], subidx++,
                    If[MemberQ[externalIndices, element["indices"][[subidx
                        ]][[2]][[1]]],
                        AppendTo[edges, element["indices"][[subidx]][[
                            2]][[1]] \[UndirectedEdge] vertexName];
                        AppendTo[vertices, element["indices"][[subidx
                            ]][[2]][[1]]];
                        AppendTo[edgesType, ToString[element["indices"
                            ][[subidx]][[1]]] <> ToString[element["indices"][[subidx]][[1]]]]
                        ,
                        AppendTo[merges, element["indices"][[subidx]]
                            [[2]][[1]] -> vertexName]
                    ];
                ];
                Clear[subidx];
            ];

            (*In case it is a Regulatordot, we have rules for merging vertices*)
            If[element["type"] == "Regulatordot",
                vertexName = RegulatorDot;
                AppendTo[vertices, vertexName];
                Clear[subidx];
                For[subidx = 1, subidx <= 2, subidx++,
                    AppendTo[merges, element["indices"][[subidx]][[2]]
                        [[1]] -> vertexName]
                ];
                Clear[subidx];
            ];
        ];
        Clear[idx];

        (*build the graph*)
        prefactor = ("Prefactor" /. diagram[[1]])[[1]];
        propagators = edges //. merges;
        style = edgesType /. styleOpts;
        edgeStyle = Map[#[[1]] -> #[[2]]&, MapThread[List, {edges, style
            }]];
        vertexShapes = Join[Map[# -> ""&, externalIndices], {RegulatorDot
             -> cross[1]}];
        vertexLabels = Map[# -> pick[derivativeList, #]&, externalIndices
            ];
        styledPropagators = MapThread[Style[#1, #2]&, {propagators, style
            }];
        {prefactor, Graph[vertices, styledPropagators, VertexShape ->
             vertexShapes, VertexSize -> {RegulatorDot -> Medium}, VertexLabels 
            -> vertexLabels]}
    ];

PlotDiagrams[setup_, derivativeList_, styleOpts_] :=
    Module[
        {diagrams, allData, allEdgeTypes, allGraphs, areListsEqual, removeDigits,
             idx, curTotal, subidx}
        ,
        (*derive the diagrams*)
        diagrams = QMeSderivation`DeriveFunctionalEquation[setup, derivativeList,
             "OutputLevel" -> "SuperindexDiagrams"];
        allGraphs = Map[DrawOneDiagram[#, derivativeList, styleOpts]&,
             diagrams];
        (* It would be nice to find all isomorphic feynman diagrams, 
            but this is actually pretty hard, so not today. *)
        allGraphs
    ];


End[];


EndPackage[];
