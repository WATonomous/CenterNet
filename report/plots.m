VALIDATION_LOSS_INTERVAL = 5;

training_loss_from_scratch = [
2.970965
2.165303
1.904277
1.760062
1.672738
1.594977
2.105640
3.204026
3.157730
3.167514
3.297638
3.204396
3.205422
3.322516
3.565203
3.495813
3.436736
3.425291
];

validation_loss_from_scratch = [
3.051442
4.342558
4.806303
];

training_loss_transfer_learning = [
2.293733
1.792962
1.783581
1.686586
1.581395
1.548590
1.487658
1.464919
1.445383
1.418578
1.415180
];

validation_loss_transfer_learning = [
2.793028
2.476969
];

num_epochs = min(numel(training_loss_from_scratch),numel(training_loss_transfer_learning));
validation_epochs = 0:VALIDATION_LOSS_INTERVAL:num_epochs;
validation_epochs = validation_epochs(2:end);
num_validation_epochs = idivide(num_epochs, int16(VALIDATION_LOSS_INTERVAL));

plot(1:num_epochs,training_loss_from_scratch(1:num_epochs),'Color','#0072BD')
hold on
plot(validation_epochs,validation_loss_from_scratch(1:num_validation_epochs),'Color','#0072BD','Marker','o','LineStyle','none')
plot(1:num_epochs,training_loss_transfer_learning(1:num_epochs),'Color','#D95319')
plot(validation_epochs,validation_loss_transfer_learning(1:num_validation_epochs),'Color','#D95319','Marker','o','LineStyle','none');
legend([
    "training loss (from scratch)"
    "validation loss (from scratch)"
    "training loss (transfer learning)"
    "validation loss (transfer learning)"
],'Location','northwest')
xlabel("Epoch")
ylabel("Loss")
hold off


