from scikit_clustering_celery import mean_shift

value = mean_shift.apply_async(args=([4.9,5.2,2.4,2.3],))
print(value.get())
