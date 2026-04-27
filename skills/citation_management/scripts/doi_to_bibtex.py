#!/usr/bin/env python3
"""通过 CrossRef API 把 DOI 转换为 BibTeX 的小工具。"""

import sys
import requests
import argparse
import time
import json
from typing import Optional, List

class DOIConverter:
    """用 CrossRef API 把 DOI 转成 BibTeX 条目。"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DOIConverter/1.0 (Citation Management Tool; mailto:support@example.com)'
        })

    def doi_to_bibtex(self, doi: str) -> Optional[str]:
        doi = doi.strip()
        if doi.startswith('https://doi.org/'):
            doi = doi.replace('https://doi.org/', '')
        elif doi.startswith('http://doi.org/'):
            doi = doi.replace('http://doi.org/', '')
        elif doi.startswith('doi:'):
            doi = doi.replace('doi:', '')

        # 通过 CrossRef 内容协商拿 BibTeX
        url = f'https://doi.org/{doi}'
        headers = {
            'Accept': 'application/x-bibtex',
            'User-Agent': 'DOIConverter/1.0 (Citation Management Tool)'
        }

        try:
            response = self.session.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                bibtex = response.text.strip()
                # CrossRef 偶尔会返回 @data 类型，统一改成 @misc
                if bibtex.startswith('@data{'):
                    bibtex = bibtex.replace('@data{', '@misc{', 1)
                return bibtex
            elif response.status_code == 404:
                print(f'Error: DOI not found: {doi}', file=sys.stderr)
                return None
            else:
                print(f'Error: Failed to retrieve BibTeX for {doi} (status {response.status_code})', file=sys.stderr)
                return None

        except requests.exceptions.Timeout:
            print(f'Error: Request timeout for DOI: {doi}', file=sys.stderr)
            return None
        except requests.exceptions.RequestException as e:
            print(f'Error: Request failed for {doi}: {e}', file=sys.stderr)
            return None

    def convert_multiple(self, dois: List[str], delay: float = 0.5) -> List[str]:
        bibtex_entries = []

        for i, doi in enumerate(dois):
            print(f'Converting DOI {i+1}/{len(dois)}: {doi}', file=sys.stderr)
            bibtex = self.doi_to_bibtex(doi)

            if bibtex:
                bibtex_entries.append(bibtex)

            # 限速，最后一次请求后不再等待
            if i < len(dois) - 1:
                time.sleep(delay)

        return bibtex_entries


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description='Convert DOIs to BibTeX format using CrossRef API',
        epilog='Example: python doi_to_bibtex.py 10.1038/s41586-021-03819-2'
    )

    parser.add_argument(
        'dois',
        nargs='*',
        help='DOI(s) to convert (can provide multiple)'
    )

    parser.add_argument(
        '-i', '--input',
        help='Input file with DOIs (one per line)'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output file for BibTeX (default: stdout)'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between requests in seconds (default: 0.5)'
    )

    parser.add_argument(
        '--format',
        choices=['bibtex', 'json'],
        default='bibtex',
        help='Output format (default: bibtex)'
    )

    args = parser.parse_args()

    dois = []

    if args.dois:
        dois.extend(args.dois)

    if args.input:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                file_dois = [line.strip() for line in f if line.strip()]
                dois.extend(file_dois)
        except FileNotFoundError:
            print(f'Error: Input file not found: {args.input}', file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f'Error reading input file: {e}', file=sys.stderr)
            sys.exit(1)

    if not dois:
        parser.print_help()
        sys.exit(1)

    converter = DOIConverter()

    if len(dois) == 1:
        bibtex = converter.doi_to_bibtex(dois[0])
        if bibtex:
            bibtex_entries = [bibtex]
        else:
            sys.exit(1)
    else:
        bibtex_entries = converter.convert_multiple(dois, delay=args.delay)

    if not bibtex_entries:
        print('Error: No successful conversions', file=sys.stderr)
        sys.exit(1)

    if args.format == 'bibtex':
        output = '\n\n'.join(bibtex_entries) + '\n'
    else:
        output = json.dumps({
            'count': len(bibtex_entries),
            'entries': bibtex_entries
        }, indent=2)

    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f'Successfully wrote {len(bibtex_entries)} entries to {args.output}', file=sys.stderr)
        except Exception as e:
            print(f'Error writing output file: {e}', file=sys.stderr)
            sys.exit(1)
    else:
        print(output)

    if len(dois) > 1:
        success_rate = len(bibtex_entries) / len(dois) * 100
        print(f'\nConverted {len(bibtex_entries)}/{len(dois)} DOIs ({success_rate:.1f}%)', file=sys.stderr)


if __name__ == '__main__':
    main()
