import click


#TODO:install conda
@click.command()
@click.option('-e', '--experiment', 'experiment', help='Experiment type (daw, dezfoulli, doll)', required=True)
def main(experiment):
    #if you import globally, compile functions will conflict
    if experiment == 'daw':
        print("Simulating Daw Task (2011)")
        from simulations.daw_task.simulation_daw_task import daw_task
        daw_task()
    elif experiment == 'daw-ventral':
        print("Simulating Daw Task with ventral loop only (2011)")
        from simulations.daw_task.simulation_daw_task_ventral_loop import daw_task_ventral
        daw_task_ventral()
    elif experiment == 'daw-ventral-optuna':
        print("Simulating Daw Task with ventral loop only and optuna parameters(2011)")
        from simulations.daw_task.simulation_daw_task_ventral_loop_optuna import daw_task_ventral_optuna
        daw_task_ventral_optuna()
    elif experiment == 'daw-ventral-fixed':
        print("Simulating Daw Task with ventral loop with fixed probs only (2011)")
        from simulations.daw_task.simulation_daw_task_ventral_fixed import daw_task_ventral_fixed
        daw_task_ventral_fixed()
    elif experiment == 'daw-dorsomedial-fixed':
        print("Simulating Daw Task with dorsomedial loop with fixed probs only (2011)")
        from simulations.daw_task.simulation_daw_task_dorsomedial import daw_task_dorsomedial
        daw_task_dorsomedial()
    elif experiment == 'dezfoulli':
        print("Simulating Dezfoulli Task (2013)")
        from simulations.dezfoulli_task.simulation_dezfoulli_task import dezfoulli_task
        dezfoulli_task()
    elif experiment == 'dezfoulli-ventral':
        print("Simulating Dezfoulli Task with ventral loop only (2013)")
        from simulations.dezfoulli_task.simulation_dezfoulli_task_ventral_loop import dezfoulli_task_ventral
        dezfoulli_task_ventral()
    elif experiment == 'dezfoulli-ventral-fixed':
        print("Simulating Dezfoulli Task with ventral loop with fixed probs only (2013)")
        from simulations.dezfoulli_task.simulation_dezfoulli_task_ventral_loop_fixed import dezfoulli_task_ventral_fixed
        dezfoulli_task_ventral_fixed()
    elif experiment == 'dezfoulli-ventral-optuna':
        print("Simulating Dezfoulli Task with ventral loop with optuna params (2013)")
        from simulations.dezfoulli_task.simulation_dezfoulli_task_ventral_loop_optuna import dezfoulli_task_ventral_optuna
        dezfoulli_task_ventral_optuna()
    elif experiment == 'doll':
        print("Simulating Doll Task (2015)")
        from simulations.doll_task.simulation_doll_task import doll_task
        doll_task()
    elif experiment == 'doll-ventral':
        print("Simulating Doll Task with ventral loop only (2015)")
        from simulations.doll_task.simulation_doll_task_ventral import doll_task_ventral
        doll_task_ventral()
    elif experiment == 'doll-ventral-fixed':
        print("Simulating Doll Task with ventral loop with fixed probs only (2015)")
        from simulations.doll_task.simulation_doll_task_ventral_fixed import doll_task_ventral_fixed
        doll_task_ventral_fixed()
    elif experiment == 'doll-dorsomedial-fixed':
        print("Simulating Doll Task with dorsomedial loop with fixed probs only (2015)")
        from simulations.doll_task.simulation_doll_task_dorsomedial import doll_task_dorsomedial
        doll_task_dorsomedial()
    else:
        print("Missing experiment parameter")

if __name__ == '__main__':
    main()