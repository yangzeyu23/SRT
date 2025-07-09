classdef vectorizingLayer < nnet.layer.Layer
   
    methods
        function layer = vectorizingLayer(name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "Collapse all the dimensions of the input into 1D vector";
            
        end
        
        function Z = predict(~, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
            [H,W,C,N] = size(X);
            Z = reshape(X,[H*W*C,1,N]);
        end
                
    end
end