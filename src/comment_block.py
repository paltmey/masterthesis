class CommentBlock:
    def __init__(self, start, end, lines):
        self.start = start
        self.end = end
        self.lines = lines

    def __repr__(self):
        return f'CommentBlock(start={self.start}, end={self.end}, lines={self.lines})'

    def __str__(self):
        lines = self.lines[:]
        lines[0] = lines[0][self.start[1]:]
        cleaned_lines = [line.strip('#\n ') for line in lines]
        filtered_lines = [line for line in cleaned_lines if line]
        merged_lines = ' '.join(filtered_lines)
        return merged_lines
