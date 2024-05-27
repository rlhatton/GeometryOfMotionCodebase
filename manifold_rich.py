#! /usr/bin/python3


class Manifold:
    """
    Class to hold manifold structure
    """

    def __init__(self, transition_table, chart_names='default'):

        # Save the provided chart transition table as a class attribute
        self.transition_table = transition_table
        # Extract the number of charts implied by the transition table
        self.n_charts = len(transition_table)

        # If the charts are not named, generate a numbered list as their names
        if chart_names != 'default':
            self.chart_names = chart_names
        else:
            self.chart_names = list(range(0, n_charts))

        # Generate a dictionary for converting a chart number to its name
        self.chart_number_to_name = dict(enumerate(self.chart_names))
        # Generate a dictionary for converting a chart name to its number
        self.chart_name_to_number = {v: k for k, v in self.chart_names.items()}

        def GetChartNumberFromNameOrNumber(self, chart_identification):

            if 

            return





class ManifoldElement:
    """
    Class for manifold elements
    """

    def __init__(self, manifold, initial_chart):

        self.manifold = manifold


