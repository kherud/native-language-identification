\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{info1thesis}[2016/01/27 Chair I thesis template]



%%% Read class options
% type of thesis
\newif\ifthesis@bachelor\thesis@bachelorfalse
\newif\ifthesis@master\thesis@masterfalse
\newif\ifthesis@seminar\thesis@seminarfalse
\newif\ifthesis@practical\thesis@practicalfalse
% used document class (scrartcl or scrreprt)
\newif\ifthesis@artcl\thesis@artclfalse
% to check if one of the options was given
\newif\ifthesis@typeset\thesis@typesetfalse
\DeclareOption{bachelor}{
    \thesis@bachelortrue
    \thesis@artclfalse
    \thesis@typesettrue
}
\DeclareOption{master}{
    \thesis@mastertrue
    \thesis@artclfalse
    \thesis@typesettrue
}
\DeclareOption{seminar}{
    \thesis@seminartrue
    \thesis@artcltrue
    \thesis@typesettrue
}
\DeclareOption{practical}{
    \thesis@practicaltrue
    \thesis@artcltrue
    \thesis@typesettrue
}

% language
\newif\ifthesis@english\thesis@englishfalse
\DeclareOption{english}{
    \thesis@englishtrue
}
\DeclareOption{german}{
    \thesis@englishfalse
}

% pass unknown options to base class
\DeclareOption*{%
    \PassOptionsToClass{\CurrentOption}{scrartcl}%
    \PassOptionsToClass{\CurrentOption}{scrreprt}%
}

% Read the class options
\ProcessOptions\relax

% Print error when no type (bachelor, master, ... was chosen)
\ifthesis@typeset\else
    \ClassError{info1thesis}{Did not select type of document. You have to
    use one of the following class options: bachelor, master, seminar,
    practical}{You have to use one of the following class options: bachelor,
    master, seminar, practical.}
\fi



%%% Load the base class
\ifthesis@artcl
    \LoadClass[a4paper,abstract,titlepage]{scrartcl}
    \newcommand{\thesis@topbookmarklevel}{1}
\else
    \LoadClass[a4paper,bibliography=totoc]{scrreprt}
    \newcommand{\thesis@topbookmarklevel}{0}
\fi



%%% Language handling
% Set language specific package options
\ifthesis@english
    \usepackage[ngerman,main=english]{babel}
    \PassOptionsToPackage{english}{algorithm2e}
    \PassOptionsToPackage{figurename=Fig.,tablename=Tab.}{caption}
\else
    \usepackage[main=ngerman]{babel}
    \PassOptionsToPackage{ngerman}{algorithm2e}
    \PassOptionsToPackage{figurename=Abb.,tablename=Tab.}{caption}
\fi

% Translations (mostly for title page)
\AtBeginDocument{%
    \providecaptionname{english}{\thesis@bachelorthesisname}{Bachelor Thesis}%
    \providecaptionname{english}{\thesis@masterthesisname}{Master Thesis}%
    \providecaptionname{english}{\thesis@seminarpapername}{Seminar Paper}%
    \providecaptionname{english}{\thesis@practicalpapername}{Practical Course Report}%
    \providecaptionname{english}{\thesis@supervisorname@pl}{Advisors}%
    \providecaptionname{english}{\thesis@supervisorname@sing}{Advisor}%
    \providecaptionname{english}{\thesis@submissiondate}{Date of Submission}%
    \providecaptionname{english}{\thesis@semester}{Semester}%
    \providecaptionname{english}{\thesis@titlepage}{Title Page}%
    %
    \providecaptionname{ngerman}{\thesis@bachelorthesisname}{Praktikumsbericht}%
    \providecaptionname{ngerman}{\thesis@masterthesisname}{Masterarbeit}%
    \providecaptionname{ngerman}{\thesis@seminarpapername}{Seminararbeit}%
    \providecaptionname{ngerman}{\thesis@practicalpapername}{Praktikumsbericht}%
    \providecaptionname{ngerman}{\thesis@supervisorname@pl}{Betreuer}%
    \providecaptionname{ngerman}{\thesis@supervisorname@sing}{Betreuer}%
    \providecaptionname{ngerman}{\thesis@submissiondate}{Abgabedatum}%
    \providecaptionname{ngerman}{\thesis@semester}{Semester}%
    \providecaptionname{ngerman}{\thesis@titlepage}{Titelseite}%
}


%%% Load some packages with our default settings
\usepackage[linesnumbered,algoruled,longend,vlined]{algorithm2e}
\DontPrintSemicolon
\SetArgSty{}
\SetKw{KwOr}{or}
\SetKw{KwAnd}{and}
\SetKw{KwNot}{not}
\setlength{\algomargin}{3ex}

\usepackage[fixlanguage]{mybabelbib}
\setbibliographyfont{title}{}
\setbibliographyfont{jtitle}{}
\setbibliographyfont{btitle}{\emph}
\setbibliographyfont{stitle}{\emph}
\setbibliographyfont{journal}{\emph}

\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{amsthm}

\usepackage{graphicx}
\usepackage[bookmarks,bookmarksnumbered,pdfencoding=auto]{hyperref}
\usepackage[font=small,format=hang,labelfont=bf]{caption}
\usepackage[labelfont=normalfont]{subcaption}
\usepackage{enumerate}

% needed for AfterEndPreamble
\RequirePackage{etoolbox}

% We use the following command to detect if there are multiple supervisors
% (to decide if we use singular or plural for the label)
% Written by Gustavo Mezzetti
% http://tex.stackexchange.com/questions/255921/check-occurrence-of-a-macro-inside-another-macro
\@ifdefinable\MacroDoesNotContainSavedName{}% Check for availability of names:
\@ifdefinable\@MacroDoesNotContainSavedName{}
\newcommand*\DefineMacroNameToBeSearched[1]{%
  \def\@MacroDoesNotContainSavedName##1#1##2\@@@{%
    \ifx\@empty##2\@empty % if ##2 is empty
  }%
  \def\MacroDoesNotContainSavedName##1{%
    TT\fi
    \expandafter\@MacroDoesNotContainSavedName##1#1\@@@
  }%
}



%%% Customized title page
% Set the subject variable depending on the selected document type
\ifthesis@bachelor
    \subject{\thesis@bachelorthesisname}
\fi
\ifthesis@master
    \subject{\thesis@masterthesisname}
\fi
\ifthesis@seminar
    \subject{\thesis@seminarpapername}
\fi
\ifthesis@practical
    \subject{\thesis@practicalpapername}
\fi

% introduce command for supervisors
\newcommand{\@supervisors}{}
\newcommand{\supervisors}[1]{%
  \gdef\@supervisors{%
    \begingroup%
      \renewcommand{\and}{\tabularnewline}%
      \begin{tabular}[t]{@{}l@{}}#1\end{tabular}%
    \endgroup%
  }%
}
\let\@germandate=\@empty
\newcommand{\germandate}[1]{\gdef\@germandate{#1}}
\newcommand{\@semester}{}
\newcommand{\semester}[1]{\gdef\@semester{#1}}

% TODO: adjust maketitle for seminar report
\if@titlepage
  \renewcommand{\maketitle}{%
    % Detect if we need singular or plural for supervisors
    \DefineMacroNameToBeSearched{\and}%
    \if\MacroDoesNotContainSavedName{\@supervisors}%
        \def\supervisorname{\thesis@supervisorname@sing}%
    \else%
        \def\supervisorname{\thesis@supervisorname@pl}%
    \fi%
    % this code is based on the definition of \maketitle in scrreprt.cls
    \begin{titlepage}
      \setlength{\parindent}{\z@}
      \setlength{\parskip}{\z@}
      \begin{center}
        \ifx\@subject\@empty \else
          {\subject@font \@subject \par}%
          \vskip 3em
        \fi
        {\titlefont\huge \@title\par}%
        \vskip 1em
        {\ifx\@subtitle\@empty\else\usekomafont{subtitle}\@subtitle\par\fi}%
        \vskip 2em
        {\Large \lineskip 0.75em
          \begin{tabular}[t]{c}
            \@author
          \end{tabular}\par
        }%
        \vskip \z@ \@plus3fill
        \begin{tabular}[t]{ll}
          \ifthesis@seminar%
            \thesis@semester: & \@semester \\%
          \fi
          \thesis@submissiondate: & 16. August 2020 \\
          \supervisorname: & \@supervisors
        \end{tabular}
        \vskip \z@ \@plus3fill
        % TODO: evtl. Auflösung des Siegels erhöhen
        \includegraphics[width=7cm]{img/siegel}
        \vskip 3em
        {
          Julius-Maximilians-Universit\"at W\"urzburg\\
          Lehrstuhl f\"ur Informatik X\\
          Data Science
        }%
      \end{center}
    \end{titlepage}%
  }
\fi
% Automatically insert title page at begin of document
\AfterEndPreamble{
    \pdfbookmark[\thesis@topbookmarklevel]{\thesis@titlepage}{title}
    % store author for later use (is deleted during maketitle)
    \global\let\thesis@author\@author
    \maketitle
}



%%% Theorem environments
\ifthesis@english
    % TODO: english theorem environments
    \ifthesis@artcl
        \newtheorem{theorem}{Theorem}
    \else
        \newtheorem{theorem}{Theorem}[chapter]
    \fi
    \theoremstyle{definition}
    \newtheorem{definition}[theorem]{Definition}
\else
    \ifthesis@artcl
        \newtheorem{satz}{Satz}
    \else
        \newtheorem{satz}{Satz}[chapter]
    \fi
    \newtheorem{lemma}[satz]{Lemma}
    \newtheorem{beobachtung}[satz]{Beobachtung}
    \newtheorem{folgerung}[satz]{Folgerung}
    \newtheorem{korollar}[satz]{Korollar}
    \theoremstyle{definition}
    \newtheorem{definition}[satz]{Definition}
\fi



%%% Abstract
\ifthesis@artcl
\else
    \let\thesis@oldabstract\abstract
    \let\endthesis@oldabstract\endabstract
    \renewenvironment{abstract}{%
        \pdfbookmark[\thesis@topbookmarklevel]{\abstractname}{abstract}
        \addchap*{\abstractname}
    }{\par}
\fi
\ifthesis@english
    \ifthesis@artcl
        \newenvironment{germanabstract}{%
            \begin{otherlanguage}{ngerman}%
                \begin{abstract}%
        }{%
                \end{abstract}%
            \end{otherlanguage}
        }
    \else
        \newenvironment{germanabstract}{%
            \begin{otherlanguage}{ngerman}%
                % Prevent page break before chapter title
                \let\clearpage\relax
                \addchap*{\abstractname}
        }{%
            \end{otherlanguage}
        }
    \fi
\else
    \newenvironment{germanabstract}{%
        \ClassError{info1thesis}{Environment 'germanabstract' should be used in
            English theses only. If you write your thesis in German, simply use
            the 'abstract' environment instead.}{}}{}
\fi



%%% Table of Contents
\newcommand{\thesistableofcontents}{
    \ifthesis@artcl
    \else
        \clearpage
        \pdfbookmark[\thesis@topbookmarklevel]{\contentsname}{toc}
        \tableofcontents
    \fi
}



%%% Bibliography
\newcommand{\thesisbibliography}{
    \clearpage
    \bibliographystyle{abbrv}
}



%%% Erklaerung
%\ifthesis@artcl
%\else
%    \AtEndDocument{
%        %%% Erklärung
%        \clearpage
%        \ifx\@germandate\@empty
%            \gdef\@germandate{\@date}
%        \fi
%        \begin{otherlanguage}{ngerman}
%            \pdfbookmark[\thesis@topbookmarklevel]{Erkl\"arung}{erklaerung}
%            \chapter*{Erkl\"arung}
%
%            \noindent
%            Hiermit versichere ich die vorliegende Abschlussarbeit
%            selbstst\"andig verfasst zu haben, keine anderen als die angegebenen
%            Quellen und Hilfsmittel benutzt zu haben, und die Arbeit bisher oder
%            gleichzeitig keiner anderen Pr\"ufungsbeh\"orde unter Erlangung
%            eines akademischen Grades vorgelegt zu haben.
%            \vspace{1cm}
%
%            \noindent
%            W\"urzburg, den \@germandate \\[1.5cm]
%            $\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots$ \\
%            \begingroup\renewcommand{\and}{\qquad}\thesis@author\endgroup
%        \end{otherlanguage}
%    }
%\fi

