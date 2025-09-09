# LaTeX Slide References Commands Guide

This guide provides comprehensive documentation for the LaTeX commands designed to format references in presentation slides with small text and separation lines.

## Quick Start

### 1. Installation

Add these files to your presentation directory:
- `slide_references_command.tex` - Contains all command definitions
- `example_presentation.tex` - Complete working example

### 2. Basic Usage

In your presentation preamble:

```latex
\documentclass{beamer}
\usepackage{xparse}  % Optional: for enhanced commands
\input{slide_references_command.tex}
```

In your slides:

```latex
\begin{frame}
    \frametitle{Your Slide Title}
    
    Your slide content here...
    
    \sliderefs{Your references here...}
\end{frame}
```

## Available Commands

### 1. `\sliderefs{text}` - Basic Command

**Purpose**: Creates a 3.5cm horizontal line followed by small reference text.

**Syntax**: `\sliderefs{reference text}`

**Example**:
```latex
\sliderefs{Weinberg (1994), Heggie+ (2020), Landau (1946), van Kampen (1955), Case (1959), Ng et al. (1999, 2004); Ng \& Bhattacharjee (2021)}
```

**Features**:
- 3.5cm horizontal line
- `\scriptsize` text (smaller than normal)
- Proper spacing before and after

### 2. `\sliderefsSmaller{text}` - Even Smaller Text

**Purpose**: Uses `\tiny` instead of `\scriptsize` for extremely small text.

**Syntax**: `\sliderefsSmaller{reference text}`

**Example**:
```latex
\sliderefsSmaller{Very detailed references that need to be extra small...}
```

### 3. `\sliderefsLong{text}` - Longer Separation Line

**Purpose**: Uses a 4cm line instead of 3.5cm.

**Syntax**: `\sliderefsLong{reference text}`

**Example**:
```latex
\sliderefsLong{References with a longer separation line...}
```

### 4. `\sliderefsBottom{text}` - Bottom-Aligned

**Purpose**: Automatically pushes references to the bottom of the slide.

**Syntax**: `\sliderefsBottom{reference text}`

**Example**:
```latex
\begin{frame}
    \frametitle{Title}
    
    Main content...
    
    \sliderefsBottom{These references will appear at the bottom}
\end{frame}
```

### 5. `\sliderefsCustom[length]{text}` - Customizable Line Length

**Purpose**: Allows custom line length (requires `xparse` package).

**Syntax**: `\sliderefsCustom[line_length]{reference text}`

**Example**:
```latex
\sliderefsCustom[5cm]{References with a 5cm separation line...}
\sliderefsCustom{References with default 3.5cm line...}
```

## Customization Options

### Text Sizes Available

| Command | Text Size | Relative Size |
|---------|-----------|---------------|
| `\sliderefs` | `\scriptsize` | Small |
| `\sliderefsSmaller` | `\tiny` | Very small |

### Line Lengths

| Command | Line Length | Customizable |
|---------|-------------|--------------|
| `\sliderefs` | 3.5cm | No |
| `\sliderefsLong` | 4.0cm | No |
| `\sliderefsCustom` | 3.5cm (default) | Yes |

### Spacing Control

All commands include built-in spacing:
- `0.5em` space before the line
- Reduced space between line and text
- `0.2em` space after references (except bottom-aligned)

## Best Practices

### 1. Consistency Across Slides

Use the same command throughout your presentation:

```latex
% Good - consistent
\sliderefs{References for slide 2}
\sliderefs{References for slide 3}
\sliderefs{References for slide 4}

% Avoid mixing unless necessary
\sliderefs{References for slide 2}
\sliderefsSmaller{References for slide 3}  % Different size
```

### 2. Appropriate Text Length

**Recommended**: Keep references concise and relevant.

```latex
% Good - concise
\sliderefs{Smith (2020), Jones \& Brown (2021), Wilson et al. (2022)}

% Avoid - too long
\sliderefs{Very long list of references that takes up too much space and distracts from the main content of the slide...}
```

### 3. Positioning Guidelines

**Regular positioning**: Use for most slides
```latex
\sliderefs{Your references...}
```

**Bottom positioning**: Use when you have varying amounts of content
```latex
\sliderefsBottom{Your references...}
```

### 4. Line Length Selection

| Slide Content | Recommended Command |
|---------------|-------------------|
| Normal content | `\sliderefs` (3.5cm) |
| Wide slides/content | `\sliderefsLong` (4cm) |
| Custom needs | `\sliderefsCustom[Xcm]` |

## Advanced Usage

### 1. Custom Styling

You can modify the commands by editing `slide_references_command.tex`:

```latex
% Example: Change line thickness
\noindent\rule{3.5cm}{0.8pt}%  % Thicker line (0.8pt instead of 0.4pt)

% Example: Change text color
{\scriptsize \textcolor{gray}{#1}}%  % Gray text
```

### 2. Multiple Reference Sections

For slides with multiple reference sections:

```latex
\begin{frame}
    \frametitle{Complex Topic}
    
    Section A content...
    \sliderefs{References for Section A}
    
    \bigskip
    
    Section B content...
    \sliderefs{References for Section B}
\end{frame}
```

### 3. Integration with Bibliography

For automated citation management:

```latex
% In preamble
\usepackage{biblatex}
\addbibresource{your_references.bib}

% In slides
\sliderefs{\cite{weinberg1994,heggie2020,landau1946}}
```

## Troubleshooting

### Problem: Line appears too close to text above

**Solution**: Add more space before the command:
```latex
Your content...

\bigskip  % or \vspace{1em}
\sliderefs{Your references...}
```

### Problem: References appear too large

**Solution**: Use the smaller text version:
```latex
\sliderefsSmaller{Your references...}
```

### Problem: Line length doesn't match slide width

**Solution**: Use custom line length:
```latex
\sliderefsCustom[5cm]{Your references...}  % Adjust as needed
```

### Problem: References interfere with slide content

**Solution**: Use bottom-aligned version:
```latex
\sliderefsBottom{Your references...}
```

### Problem: Command not working

**Check**:
1. `slide_references_command.tex` is in the correct directory
2. `\input{slide_references_command.tex}` is in your preamble
3. For `\sliderefsCustom`, ensure `\usepackage{xparse}` is included

## Complete Example

Here's a minimal working example:

```latex
\documentclass{beamer}
\usepackage{xparse}
\input{slide_references_command.tex}

\title{Your Presentation}
\author{Your Name}

\begin{document}

\begin{frame}
    \titlepage
\end{frame}

\begin{frame}
    \frametitle{Introduction}
    
    \begin{itemize}
        \item Point 1
        \item Point 2  
        \item Point 3
    \end{itemize}
    
    \sliderefs{Weinberg (1994), Heggie+ (2020), Landau (1946)}
\end{frame}

\begin{frame}
    \frametitle{Results}
    
    Main results content...
    
    \sliderefsBottom{References automatically at bottom}
\end{frame}

\end{document}
```

## File Structure

Your presentation directory should contain:

```
your_presentation/
├── slide_references_command.tex    # Command definitions
├── your_presentation.tex           # Your main presentation file
└── example_presentation.tex        # Complete example (optional)
```

## Quick Reference Card

| Need | Use |
|------|-----|
| Basic references | `\sliderefs{text}` |
| Very small text | `\sliderefsSmaller{text}` |
| Longer line | `\sliderefsLong{text}` |
| Bottom of slide | `\sliderefsBottom{text}` |
| Custom line length | `\sliderefsCustom[Xcm]{text}` |

---

**Questions or Issues?** Check the `example_presentation.tex` file for complete working examples of all commands.
